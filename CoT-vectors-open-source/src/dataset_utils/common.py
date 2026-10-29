"""Shared split selection, Self-CoT loading, prompts, batching and evaluation."""
import hashlib
import json
import logging
from functools import partial
from pathlib import Path

from .answers import answers_equal
from .features import training_features
from .interface import DatasetBase

logger = logging.getLogger(__name__)


def parquet_splits(directory, support="train", query="test"):
    from datasets import load_dataset

    directory = Path(directory)
    files = {}
    for name, split in (("train", support), ("test", query)):
        matches = sorted(directory.glob(f"{split}-*.parquet"))
        if not matches:
            matches = sorted(directory.glob(f"{split}.parquet"))
        if not matches:
            raise FileNotFoundError(f"No {split} parquet files in {directory}")
        files[name] = [str(path) for path in matches]
    dataset = load_dataset("parquet", data_files=files)
    return dataset["train"], dataset["test"]


def json_records(path):
    text = Path(path).read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"Empty Self-CoT file: {path}")
    records = json.loads(text) if text.startswith("[") else [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError("Self-CoT input must contain JSON objects (JSONL or a JSON array)")
    return records


def selected_split(dataset, count, seed):
    if count is not None and count < 0:
        raise ValueError("Sample counts cannot be negative")
    dataset = dataset.shuffle(seed=seed)
    return dataset.select(range(min(count, len(dataset)))) if count else dataset


def formatted_choices(options):
    if isinstance(options, str):
        return options
    if isinstance(options, dict):
        labels, texts = options["label"], options["text"]
        if len(labels) != len(texts):
            raise ValueError("Choice labels and texts have different lengths")
    else:
        texts = options
        labels = [chr(65 + index) for index in range(len(texts))]
    return "; ".join(f"{label}. {text}" for label, text in zip(labels, texts))


class ReasoningDataset(DatasetBase):
    question_key = "question"
    answer_key = "answer"
    cot_key = "cot_content"
    requires_self_cot = False
    eval_system_prompt = ""
    teacher_system_prompt = ""
    student_system_prompt = ""

    def __init__(self, data_cfg, model_processor=None, model_name=None):
        super().__init__(data_cfg, model_processor, model_name)
        self.stage = getattr(data_cfg, "stage", "eval")
        if self.stage not in ("acquire", "eval", "prepare"):
            raise ValueError(f"Unknown dataset stage: {self.stage}")
        local_path = getattr(data_cfg, "local_path", None)
        if not local_path or not Path(local_path).is_dir():
            raise FileNotFoundError(f"data.local_path must be an existing dataset directory: {local_path}")
        self._support_set, self._query_set = self._load_splits(local_path)
        self.self_cot_path = None
        if self.stage == "acquire":
            use_self_cot = getattr(data_cfg, "use_self_cot", False)
            if self.requires_self_cot and not use_self_cot:
                raise ValueError(f"{self.name} acquisition requires model-generated Self-CoT")
            if use_self_cot:
                self._load_self_cot_data(data_cfg)
            self._support_set = selected_split(self._support_set, data_cfg.num_query_samples, data_cfg.seed)
        else:
            self._support_set = selected_split(self._support_set, 0, data_cfg.seed)
        query_count = data_cfg.num_query_samples if self.stage == "eval" else 0
        self._query_set = selected_split(self._query_set, query_count, data_cfg.seed)
        self._validation_set = self._query_set.select(range(min(16, len(self._query_set))))
        if not len(self._support_set) or not len(self._query_set):
            raise ValueError(f"{self.name}: support and query splits must be non-empty")

    def _load_splits(self, local_path):
        return parquet_splits(local_path)

    def question(self, item):
        return str(item[self.question_key])

    def ground_truth(self, item):
        answer = self.extract_answer(item[self.answer_key])
        if answer is None or not str(answer).strip():
            raise ValueError(f"{self.name}: missing/invalid ground-truth answer")
        return answer

    def _load_self_cot_data(self, data_cfg):
        from datasets import Dataset

        path = Path(getattr(data_cfg, "self_cot_path", "") or "")
        if not path.is_file():
            raise FileNotFoundError(f"Self-CoT file not found: {path}")
        by_question = {}
        for row in json_records(path):
            question, cot = row.get("question"), row.get("self_cot")
            if not isinstance(question, str) or not isinstance(cot, str) or not cot.strip():
                raise ValueError("Self-CoT records require non-empty question and self_cot strings")
            correct = row.get("is_correct")
            if correct is None:
                if "correct_only" not in path.stem:
                    raise ValueError("Self-CoT records require is_correct, unless explicitly a correct_only file")
                correct = True
            if isinstance(correct, str):
                if correct.lower() not in ("true", "false"):
                    raise ValueError("is_correct must be a boolean")
                correct = correct.lower() == "true"
            if correct is not True and correct is not False:
                raise ValueError("is_correct must be a boolean")
            if correct:
                by_question[question] = row
        samples = []
        for sample in self._support_set:
            # Join against support only. Test questions are never introduced here.
            row = by_question.get(str(sample[self.question_key]))
            if row is None:
                continue
            target = self.ground_truth(sample)
            recorded_target = row.get("gt_numerical", row.get("gt_answer"))
            if recorded_target is not None and self.extract_answer(recorded_target) != target:
                raise ValueError(f"Self-CoT ground truth disagrees with the support dataset: {row['question']}")
            samples.append({**sample, "self_cot": row["self_cot"]})
        if not samples:
            raise ValueError("No correct Self-CoT records matched the support split")
        self._support_set = Dataset.from_list(samples)
        self.self_cot_path = path

    def describe(self):
        return {"dataset": self.name, "stage": self.stage, "seed": self.cfg.seed,
                "support_samples": len(self._support_set), "query_samples": len(self._query_set),
                "self_cot_path": str(self.self_cot_path) if self.self_cot_path else None,
                "self_cot_sha256": hashlib.sha256(self.self_cot_path.read_bytes()).hexdigest() if self.self_cot_path else None}

    def messages(self, item, teacher=False, direct=False):
        prompt = self.teacher_system_prompt if teacher else self.student_system_prompt if direct else self.eval_system_prompt
        messages = [{"role": "system", "content": prompt},
                    {"role": "user", "content": self.question(item)}]
        if teacher:
            answer = self.ground_truth(item)
            response = str(item.get("self_cot", item.get(self.cot_key, "")))
            if not response.strip():
                raise ValueError(f"{self.name}: no CoT response for acquisition")
            # Guarantee an unambiguous final answer while preserving complete CoT traces.
            if self.extract_answer(response) != answer or str(answer) not in response:
                response += "\n" + r"\boxed{" + str(answer) + "}"
            messages.append({"role": "assistant", "content": response})
        return messages

    def training_examples(self, batch_data):
        if self.model_processor is None:
            raise ValueError("A tokenizer is required to construct training inputs")
        examples = []
        for item in batch_data:
            teacher = self.messages(item, teacher=True)
            student = self.messages(item, direct=True)
            examples.append({
                "teacher_text": self.model_processor.apply_chat_template(teacher, tokenize=False, add_generation_prompt=False),
                "response": teacher[-1]["content"], "answer": self.ground_truth(item),
                "student_text": self.model_processor.apply_chat_template(student, tokenize=False, add_generation_prompt=True)})
        return examples

    def collate_fn_for_train(self, batch_data, max_seq_len):
        import torch

        features = training_features(self.training_examples(batch_data), self.model_processor, max_seq_len)
        for key in ("prefix_inputs", "student_inputs"):
            features[key] = {name: torch.tensor(value, dtype=torch.long) for name, value in features[key].items()}
        for key in ("teacher_answer_mask", "teacher_cot_mask"):
            features[key] = torch.tensor(features[key], dtype=torch.bool)
        features["student_labels"] = torch.tensor(features["student_labels"], dtype=torch.long)
        return features

    def train_dataloader(self, model, batch_size, distributed=True):
        from torch.utils.data import DataLoader, DistributedSampler

        if self.stage != "acquire":
            raise ValueError("Training/extraction requires data.stage=acquire")
        self.model_processor = model.processor
        sampler = DistributedSampler(self._support_set, shuffle=True, seed=self.cfg.seed) if distributed else None
        return DataLoader(self._support_set, batch_size=batch_size, sampler=sampler,
                          shuffle=sampler is None, num_workers=getattr(self.cfg, "num_workers", 0),
                          collate_fn=partial(self.collate_fn_for_train, max_seq_len=getattr(self.cfg, "max_seq_len", 2048)))

    def collate_fn_for_validation(self, batch_data):
        return {"batch_messages": [self.messages(item) for item in batch_data],
                "ground_truths": [self.ground_truth(item) for item in batch_data]}

    def validation_dataloader(self, batch_size):
        from torch.utils.data import DataLoader

        return DataLoader(self._validation_set, batch_size=batch_size, shuffle=False,
                          collate_fn=self.collate_fn_for_validation)

    def eval(self, eval_cfg, model):
        if self.stage != "eval":
            raise ValueError("Evaluation requires data.stage=eval")
        if getattr(eval_cfg.data, "num_shot", 0):
            raise ValueError("The main experiments use zero-shot evaluation")
        requested = getattr(eval_cfg, "max_samples", None)
        if requested and requested > len(self._query_set):
            raise ValueError(f"Requested {requested} evaluation samples, only {len(self._query_set)} selected")
        samples = self._query_set.select(range(requested)) if requested else self._query_set
        records = []
        for start in range(0, len(samples), eval_cfg.batch_size):
            batch = [samples[index] for index in range(start, min(start + eval_cfg.batch_size, len(samples)))]
            prompts = [model.processor.apply_chat_template(self.messages(item), tokenize=False, add_generation_prompt=True) for item in batch]
            # Fail immediately on OOM/invalid output. Never reuse or silently skip predictions.
            predictions = model.generate(prompts, **eval_cfg.generation_args)
            if predictions is None or len(predictions) != len(batch):
                raise RuntimeError("Generation did not return one prediction per example")
            for item, prediction in zip(batch, predictions):
                extracted = self.extract_answer(prediction)
                target = self.ground_truth(item)
                records.append({"question": str(item[self.question_key]), "model_output": prediction,
                                "extracted_prediction": extracted, "ground_truth": str(item[self.answer_key]),
                                "extracted_ground_truth": target, "is_correct": answers_equal(extracted, target)})
        if not records:
            raise ValueError("The evaluation split is empty")
        return records, {"accuracy": sum(row["is_correct"] for row in records) / len(records)}
