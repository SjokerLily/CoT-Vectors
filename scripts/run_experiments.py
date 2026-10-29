"""Main experiment orchestration only; model, hook, loss and scoring stay in src/."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"qwen2.5-math-7b": 28, "llama-3.1-8b-instruct": 32}
TASKS = {"gsm8k": ("gsm8k", None), "math-easy": ("math_dataset", "easy"),
         "math-hard": ("math_dataset", "hard"), "mmlu-pro": ("mmlu-pro", None),
         "commonsenseqa": ("commonsenseqa", None), "strategyqa": ("strategyqa", None)}
SELF_COT_TASKS = {"commonsenseqa", "strategyqa"}

def override(key, value):
    # JSON strings are also Hydra quoted values; subprocess receives an argv list.
    return key + "=" + json.dumps(value, ensure_ascii=False)

def layers_from(value, count):
    layers = list(range(count)) if value == "all" else [int(x) for x in value.split(",")]
    if not layers or len(set(layers)) != len(layers) or any(x < 0 or x >= count for x in layers):
        raise ValueError(f"layers must be unique indices in [0, {count - 1}]")
    return layers

def support_size(action, model, task):
    if task in SELF_COT_TASKS:
        return 500 if action == "extract" else 0
    if task == "mmlu-pro":
        return 70
    if action == "extract":
        return 3000
    if model.startswith("qwen"):
        return 3000 if task == "gsm8k" else 2000
    return 2000 if task == "gsm8k" else 1000

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def summarize(directory, layers, count, expected_samples):
    (directory / "summary.json").unlink(missing_ok=True)
    scores = []
    for layer in layers:
        file = directory / f"layer-{layer}.json"
        result = read_json(file)
        records = result["records"]
        if len(records) != expected_samples:
            raise ValueError(f"{file}: expected {expected_samples} evaluated samples, got {len(records)}")
        accuracy = result["eval_result"]["accuracy"]
        if not isinstance(accuracy, (int, float)) or not 0 <= accuracy <= 1:
            raise ValueError(f"Invalid accuracy in {file}")
        scores.append({"layer": layer, "accuracy": accuracy, "samples": len(records), "file": str(file)})
    best = max(scores, key=lambda row: row["accuracy"])
    summary = {"layers": scores, "best_layer": best["layer"], "best_accuracy": best["accuracy"],
               "full_layer_scan": set(layers) == set(range(count)),
               "selection": "maximum evaluation accuracy over requested independent single-layer runs"}
    write_json(directory / "summary.json", summary)
    print(f"Best layer: {best['layer']}; accuracy: {100 * best['accuracy']:.2f}%")
    return summary

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["extract", "learn", "eval-extracted", "eval-learned"])
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--dataset", required=True, choices=TASKS)
    parser.add_argument("--model-path", required=True, type=Path, help="Local Hugging Face model directory")
    parser.add_argument("--data-path", required=True, type=Path, help="Local parquet / saved MATH dataset directory")
    parser.add_argument("--output-dir", type=Path, help="Use the same directory for acquisition and evaluation")
    parser.add_argument("--layers", default="all", help="all, a single index, or comma-separated indices (each run inserts ONE layer)")
    parser.add_argument("--devices", default="0", help="CUDA_VISIBLE_DEVICES; evaluation/extraction use its first GPU")
    parser.add_argument("--support-samples", type=int, help="Override acquisition size; 0 uses all correct support records")
    parser.add_argument("--self-cot-path", type=Path, help="Support-only Self-CoT JSONL; required to acquire CSQA/SQA vectors")
    parser.add_argument("--eval-samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1, help="Learning epochs; evaluation reads this from acquisition metadata")
    parser.add_argument("--lr", type=float, help="Learning rate; default Qwen layers 0-3: .005, others: .0001")
    parser.add_argument("--warmup", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, help="Default extraction 1, learning 2, evaluation 16")
    parser.add_argument("--strategy", default="deepspeed_stage_2_offload")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without loading models or writing artifacts")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly allow replacing outputs for the selected layers")
    parser.add_argument("--summarize-only", action="store_true", help="Only summarize existing evaluation results")
    a = parser.parse_args(argv)
    if not a.devices or any(not part.isdigit() for part in a.devices.split(",")):
        parser.error("--devices must be GPU indices such as 0 or 0,1; these entry points target GPU servers")
    if a.epochs < 1 or a.eval_samples < 1 or (a.support_samples is not None and a.support_samples < 0):
        parser.error("epochs/evaluation samples must be positive; support samples must be non-negative")
    if a.batch_size is not None and a.batch_size < 1:
        parser.error("batch size must be positive")
    if a.action == "extract" and a.batch_size not in (None, 1):
        parser.error("extraction batch size must be 1 to retain per-instance averaging")
    if a.lr is not None and a.lr <= 0:
        parser.error("learning rate must be positive")
    if a.warmup is not None and not 0 <= a.warmup <= 1:
        parser.error("warmup must be between 0 and 1")
    try:
        layers = layers_from(a.layers, MODELS[a.model])
    except ValueError as error:
        parser.error(str(error))
    if a.summarize_only and not a.action.startswith("eval-"):
        parser.error("--summarize-only requires an evaluation action")
    acquiring = a.action in ("extract", "learn")
    if acquiring and a.dataset in SELF_COT_TASKS and not a.self_cot_path:
        parser.error("--self-cot-path is required for CommonsenseQA/StrategyQA acquisition")
    if a.self_cot_path:
        a.self_cot_path = a.self_cot_path.resolve()
        if not a.dry_run and not a.self_cot_path.is_file():
            parser.error(f"Self-CoT file not found: {a.self_cot_path}")
    a.model_path = a.model_path.resolve()
    a.data_path = a.data_path.resolve()
    out = (a.output_dir or ROOT / "results" / "main_experiments" / a.model / a.dataset).resolve()
    method = "extracted" if a.action in ("extract", "eval-extracted") else "learned"
    eval_dir = out / "evaluation" / method
    if a.summarize_only:
        summarize(eval_dir, layers, MODELS[a.model], a.eval_samples)
        return
    if not a.dry_run:
        for path in (a.model_path, a.data_path):
            if not path.is_dir():
                parser.error(f"directory not found: {path}")
        config = read_json(a.model_path / "config.json")
        if config["num_hidden_layers"] != MODELS[a.model]:
            parser.error("model config layer count does not match --model")
    dataset, subset = TASKS[a.dataset]
    common = [override("model_name", a.model), override("model_path", str(a.model_path)),
              override("data.name", dataset), override("data.local_path", str(a.data_path)),
              override("data.subset", subset), override("data.seed", a.seed), "hydra.job.chdir=false"]
    use_self_cot = acquiring and a.self_cot_path is not None
    if acquiring:
        common += [override("data.use_self_cot", use_self_cot),
                   override("data.self_cot_path", str(a.self_cot_path) if use_self_cot else None),
                   override("data.training_mode", "TRAIN_STUDENT_DIRECT_Q_SELF_COT" if use_self_cot else "TRAIN_STUDENT_DIRECT_Q")]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.devices,
               PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    identity = {"model": a.model, "dataset": a.dataset, "model_path": str(a.model_path),
                "data_path": str(a.data_path)}

    def execute(script, config_name, args, target, metadata_path, metadata):
        command = [sys.executable, str(ROOT / script), "--config-name", config_name, *common, *args]
        print(shlex.join(command), flush=True)
        if a.dry_run:
            return
        if (target.exists() or metadata_path.exists()) and not a.overwrite:
            raise FileExistsError(f"Output exists: {target}; use --overwrite or a different --output-dir")
        if a.overwrite:
            target.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        if a.action.startswith("eval-"):
            (eval_dir / "summary.json").unlink(missing_ok=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        if not target.exists():
            raise FileNotFoundError(f"Command exited without producing {target}")
        write_json(metadata_path, {**identity, **metadata, "command": command})

    if a.action == "extract":
        samples = a.support_samples if a.support_samples is not None else support_size("extract", a.model, a.dataset)
        bank = out / "extracted" / "vectors.pt"
        execute("src/extract_cot_vector.py", "extract",
                [override("output_path", str(bank)), override("extraction_layers", layers),
                 override("data.num_query_samples", samples), "batch_size=1"],
                bank, bank.with_suffix(".json"), {"layers": layers, "support_samples": samples, "seed": a.seed})
        return
    for layer in layers:
        run_dir = out / "learned" / f"layer-{layer}"
        manifest_path = run_dir / "manifest.json"
        if a.action == "learn":
            samples = a.support_samples if a.support_samples is not None else support_size("learn", a.model, a.dataset)
            checkpoint = run_dir / "ckpt" / f"main_experiments-{a.model}-{dataset}-{samples}" / f"epoch-{a.epochs - 1}"
            env["COT_RESULTS_DIR"] = str(run_dir)
            lr = a.lr if a.lr is not None else (0.005 if a.model.startswith("qwen") and layer < 4 else 0.0001)
            warmup = a.warmup if a.warmup is not None else (0.1 if a.dataset == "mmlu-pro" or (a.model.startswith("qwen") and a.dataset == "gsm8k") else 0.5)
            execute("src/train.py", "train",
                    ["runname=main_experiments", override("only_shift_at_layer", layer), override("epochs", a.epochs),
                     override("data.num_query_samples", samples), override("lr", lr), override("warmup_step", warmup),
                     override("batch_size", a.batch_size or 2), override("strategy", a.strategy)],
                    checkpoint / "encoder.pth", manifest_path,
                    {"layer": layer, "checkpoint": str(checkpoint), "epochs": a.epochs,
                     "support_samples": samples, "lr": lr, "warmup": warmup, "seed": a.seed})
            continue
        env["COT_RESULTS_DIR"] = str(out)
        result_file = eval_dir / f"layer-{layer}.json"
        args = [override("only_shift_at_layer", layer), override("data.num_query_samples", a.eval_samples),
                override("max_samples", a.eval_samples), override("record_path", str(result_file)),
                override("batch_size", a.batch_size or 16)]
        acquisition = out / "extracted" / "vectors.json" if method == "extracted" else manifest_path
        metadata = None
        if not a.dry_run:
            metadata = read_json(acquisition)
            if any(metadata.get(k) != v for k, v in identity.items()):
                raise ValueError(f"Acquisition model/dataset/path mismatch: {acquisition}")
            if method == "extracted" and layer not in metadata["layers"]:
                raise ValueError(f"Layer {layer} was not extracted")
            if method == "learned" and metadata["layer"] != layer:
                raise ValueError("Learning/evaluation layer mismatch")
        if method == "extracted":
            if not a.dry_run and not (out / "extracted" / "vectors.pt").is_file():
                raise FileNotFoundError(out / "extracted" / "vectors.pt")
            args += ["use_extracted_cot_vector=true", "ckpt_path=null",
                     override("extracted_cot_vector_path", str(out / "extracted" / "vectors.pt"))]
        else:
            checkpoint = metadata["checkpoint"] if metadata else str(run_dir / "ckpt" / "<from-manifest>" / "epoch-N")
            if not a.dry_run and not (Path(checkpoint) / "encoder.pth").is_file():
                raise FileNotFoundError(checkpoint)
            args += ["use_extracted_cot_vector=false", override("ckpt_path", checkpoint)]
        execute("src/eval.py", "eval", args, result_file, result_file.with_suffix(".run.json"),
                {"layer": layer, "acquisition": str(acquisition), "eval_samples": a.eval_samples})
    if a.action.startswith("eval-") and not a.dry_run:
        summarize(eval_dir, layers, MODELS[a.model], a.eval_samples)

if __name__ == "__main__":
    main()
