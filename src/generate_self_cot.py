"""Prepare support-only Self-CoT JSONL for CommonsenseQA or StrategyQA."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from src.dataset_utils import dataset_mapping


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("qwen2.5-math-7b", "llama-3.1-8b-instruct"))
    parser.add_argument("--dataset", required=True, choices=("commonsenseqa", "strategyqa"))
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--devices", default="0")
    parser.add_argument("--samples", type=int, default=0, help="0 processes the entire support split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.samples < 0 or not args.devices or any(not device.isdigit() for device in args.devices.split(",")):
        parser.error("samples must be non-negative and devices must contain CUDA indices")
    if args.output.suffix not in (".json", ".jsonl"):
        parser.error("--output must be a .json or .jsonl path (contents are JSONL)")
    correct_path = args.output.with_name(args.output.stem + "_correct_only" + args.output.suffix)
    for path in (args.output, correct_path):
        if path.exists() and not args.overwrite:
            parser.error(f"Output already exists: {path}; use --overwrite")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    import torch
    from src.utils import load_model
    from src.dataset_utils.common import selected_split

    if not torch.cuda.is_available():
        raise RuntimeError("Self-CoT generation requires a CUDA GPU")
    torch.manual_seed(args.seed)
    cfg = SimpleNamespace(model_name=args.model, model_path=str(args.model_path), dtype="torch.float16")
    model = load_model(cfg, device=torch.device("cuda:0"))
    data = SimpleNamespace(name=args.dataset, local_path=str(args.data_path), stage="prepare",
                           num_query_samples=0, seed=args.seed)
    dataset = dataset_mapping[args.dataset](data, model_name=args.model)
    support = selected_split(dataset.support_set, args.samples, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    correct_temporary = correct_path.with_suffix(correct_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as all_file, correct_temporary.open("w", encoding="utf-8") as correct_file:
        for sample in support:
            prompt = model.processor.apply_chat_template(dataset.messages(sample), tokenize=False, add_generation_prompt=True)
            outputs = model.generate([prompt], num_beams=3, max_new_tokens=512, length_penalty=0.0,
                                     do_sample=False, temperature=1.0, top_p=1.0)
            if outputs is None or len(outputs) != 1:
                raise RuntimeError("Self-CoT generation returned no response")
            answer = dataset.ground_truth(sample)
            predicted = dataset.extract_answer(outputs[0])
            record = {"question": str(sample["question"]), "gt_answer": answer, "gt_numerical": answer,
                      "self_cot": outputs[0], "predicted_answer": predicted, "is_correct": predicted == answer}
            if "choices" in sample:
                record["options"] = sample["choices"]
            line = json.dumps(record, ensure_ascii=False) + "\n"
            all_file.write(line)
            if record["is_correct"]:
                correct_file.write(line)
    temporary.replace(args.output)
    correct_temporary.replace(correct_path)
    args.output.with_suffix(args.output.suffix + ".meta.json").write_text(
        json.dumps({"model": args.model, "model_path": str(args.model_path.resolve()),
                    "data_path": str(args.data_path.resolve()), "seed": args.seed,
                    "samples": len(support), "source": "support split only"}, indent=2), encoding="utf-8")
    print(f"Saved Self-CoT to {args.output}; correct support responses to {correct_path}")


if __name__ == "__main__":
    main()
