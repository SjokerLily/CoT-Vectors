"""Pure data-pipeline checks; no PyTorch, model weights or GPU dependency."""
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.dataset_utils import dataset_mapping
from src.dataset_utils.answers import answers_equal, boolean_answer, boxed_answer, choice_answer
from src.dataset_utils.common import json_records, parquet_splits, selected_split
from src.dataset_utils.features import training_features

class MemoryDataset:
    def __init__(self, rows):
        self.rows = list(rows)
    @classmethod
    def from_list(cls, rows):
        return cls(rows)
    def __len__(self):
        return len(self.rows)
    def __iter__(self):
        return iter(self.rows)
    def __getitem__(self, index):
        return self.rows[index]
    def shuffle(self, seed):
        rows = list(self.rows)
        random.Random(seed).shuffle(rows)
        return type(self)(rows)
    def select(self, indices):
        return type(self)(self.rows[index] for index in indices)

class Tokenizer:
    pad_token_id = eos_token_id = 0
    def __call__(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("Chat templates already contain special tokens")
        return SimpleNamespace(input_ids=[ord(c) + 1 for c in text])
    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(message["content"] for message in messages)

class DatasetTests(unittest.TestCase):
    def test_answer_parsers(self):
        self.assertEqual(boxed_answer(r"first \boxed{1}; final \boxed{\frac{1}{2}}"), r"\frac{1}{2}")
        self.assertEqual(boolean_answer(False), "False")
        self.assertEqual(boolean_answer(r"\boxed{False}"), "False")
        self.assertEqual(boolean_answer("the answer is yes"), "True")
        self.assertEqual(choice_answer("The answer is J."), "J")
        self.assertIsNone(choice_answer("J", choices="ABCDE"))
        self.assertFalse(answers_equal(None, None))
        self.assertFalse(answers_equal("", ""))

    def test_selection_is_local_and_zero_means_all(self):
        data = MemoryDataset([{"id": index} for index in range(8)])
        before = random.getstate()
        self.assertEqual(list(selected_split(data, 3, 42)), list(selected_split(data, 3, 42)))
        self.assertEqual(len(selected_split(data, 0, 42)), 8)
        self.assertEqual(len(selected_split(data, 1000, 42)), 8)
        self.assertEqual(random.getstate(), before)
        with self.assertRaises(ValueError):
            selected_split(data, -1, 42)

    def test_jsonl_array_and_bom(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "support.json"
            row = {"question": "q", "self_cot": "a", "is_correct": True}
            path.write_text("\ufeff\n" + json.dumps(row) + "\n\n", encoding="utf-8")
            self.assertEqual(json_records(path), [row])
            path.write_text(json.dumps([row]), encoding="utf-8")
            self.assertEqual(json_records(path), [row])
            path.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                json_records(path)

    def dataset(self, folder, name="gsm8k", stage="eval", count=2, **kwargs):
        cls = dataset_mapping[name]
        support = MemoryDataset([{"question": f"support{i}", "answer": "work #### 2"} for i in range(5)])
        query = MemoryDataset([{"question": f"test{i}", "answer": "work #### 2"} for i in range(4)])
        cfg = SimpleNamespace(name=name, local_path=str(folder), stage=stage,
                              num_query_samples=count, seed=42, use_self_cot=False, **kwargs)
        with patch.object(cls, "_load_splits", return_value=(support, query)):
            instance = cls(cfg, model_processor=Tokenizer())
        return instance, cfg

    def test_stage_counts_and_config_immutability(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset, cfg = self.dataset(folder)
            self.assertEqual(len(dataset.support_set), 5)
            self.assertEqual(len(dataset.query_set), 2)
            self.assertEqual(cfg.num_query_samples, 2)
            dataset, cfg = self.dataset(folder, stage="acquire", count=100)
            self.assertEqual(len(dataset.support_set), 5)
            self.assertEqual(len(dataset.query_set), 4)
            self.assertEqual(cfg.num_query_samples, 100)

    def test_evaluation_count_oom_and_invalid_prediction(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset, _ = self.dataset(folder)
            cfg = SimpleNamespace(data=SimpleNamespace(num_shot=0), max_samples=2,
                                  batch_size=1, generation_args={})
            model = SimpleNamespace(processor=Tokenizer(), generate=lambda prompts, **kw: ["2"] * len(prompts))
            records, result = dataset.eval(cfg, model)
            self.assertEqual(len(records), 2)
            self.assertEqual(result["accuracy"], 1.0)
            def fail(*args, **kwargs):
                raise RuntimeError("CUDA out of memory")
            model.generate = fail
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                dataset.eval(cfg, model)
            model.generate = lambda *args, **kw: None
            with self.assertRaises(RuntimeError):
                dataset.eval(cfg, model)
            cfg.max_samples = 3
            with self.assertRaises(ValueError):
                dataset.eval(cfg, model)

    def test_self_cot_joins_support_and_rejects_false_flags(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "support_correct_only.json"
            rows = [
                {"question": "train0", "gt_answer": "False", "self_cot": "reason; False", "is_correct": True},
                {"question": "train1", "gt_answer": "True", "self_cot": "reason; True", "is_correct": False},
                {"question": "test0", "gt_answer": "True", "self_cot": "reason; True", "is_correct": True},
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            cls = dataset_mapping["strategyqa"]
            cfg = SimpleNamespace(name="strategyqa", local_path=folder, stage="acquire",
                                  num_query_samples=0, seed=42, use_self_cot=True, self_cot_path=str(path))
            support = MemoryDataset([{"question": "train0", "answer": False},
                                     {"question": "train1", "answer": True}])
            query = MemoryDataset([{"question": "test0", "answer": True}])
            with patch.dict(sys.modules, {"datasets": SimpleNamespace(Dataset=MemoryDataset)}), patch.object(cls, "_load_splits", return_value=(support, query)):
                dataset = cls(cfg)
                self.assertEqual(len(dataset.support_set), 1)
                self.assertEqual(dataset.support_set[0]["question"], "train0")
                rows[0]["is_correct"] = False
                path.write_text(json.dumps(rows), encoding="utf-8")
                with self.assertRaises(ValueError):
                    cls(cfg)

    def test_parquet_globs_support_shards_and_labeled_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ["train-00000-hash.parquet", "train-00001-hash.parquet", "validation-00000-hash.parquet"]:
                (root / name).touch()
            calls = []
            def loader(kind, **kwargs):
                calls.append((kind, kwargs["data_files"]))
                return {"train": "support", "test": "query"}
            with patch.dict(sys.modules, {"datasets": SimpleNamespace(load_dataset=loader)}):
                self.assertEqual(parquet_splits(root, query="validation"), ("support", "query"))
                self.assertEqual(len(calls[0][1]["train"]), 2)
                self.assertIn("validation", calls[0][1]["test"][0])
                with self.assertRaises(FileNotFoundError):
                    parquet_splits(root, query="test")

    def test_features_keep_answer_and_mask_after_truncation(self):
        tokenizer = Tokenizer()
        example = {"teacher_text": "question 42\nreasoning. final 42!", "response": "reasoning. final 42",
                   "answer": "42", "student_text": "long question 42\nanswer:"}
        features = training_features([example], tokenizer, max_seq_len=12)
        teacher = [token for token, flag in zip(features["prefix_inputs"]["input_ids"][0], features["teacher_answer_mask"][0]) if flag]
        student = [label for label in features["student_labels"][0] if label != -100]
        self.assertEqual(teacher, student)
        self.assertEqual(teacher, tokenizer("42").input_ids)
        self.assertEqual(len(features["prefix_inputs"]["input_ids"][0]), 12)
        self.assertEqual(features["student_inputs"]["input_ids"][0][-1], tokenizer.eos_token_id)
        self.assertEqual(features["student_inputs"]["attention_mask"][0][-1], 1)

if __name__ == "__main__":
    unittest.main()
