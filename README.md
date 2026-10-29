# CoT Vectors

Official open-source implementation of [**CoT Vectors: Transferring and Probing the Reasoning Mechanisms of LLMs**](https://arxiv.org/abs/2510.00579).

The main experiments have two stages: obtain a dataset-specific CoT Vector through **extraction** or **learning**, then **insert and evaluate** the vector at a single model layer. The scripts support both individual layers and full layer sweeps.

## Installation

Requirements: Linux, Python 3.10, and an NVIDIA GPU.

```bash
conda create -n cot-vectors python=3.10 -y
conda activate cot-vectors
pip install -r requirements.txt
```

## Models and Data

Provide a local Hugging Face model directory containing `config.json`, tokenizer files, and model weights.

| `--model` | Model | Layer indices |
| --- | --- | --- |
| `qwen2.5-math-7b` | Qwen2.5-Math-7B | 0–27 |
| `llama-3.1-8b-instruct` | Llama-3.1-8B-Instruct | 0–31 |

Use `pretraining_tp=1` for Llama.

Set `--data-path` to the root directory of one dataset:

| `--dataset` | Support split | Evaluation split | Required files or directories |
| --- | --- | --- | --- |
| `gsm8k` | train | test | `train-*.parquet`, `test-*.parquet` |
| `math-easy` | train, Levels 1–3 | test, Levels 1–3 | `math_train_subset_easy/`, `math_test_subset_easy/` |
| `math-hard` | train, Levels 4–5 | test, Levels 4–5 | `math_train_subset_hard/`, `math_test_subset_hard/` |
| `mmlu-pro` | validation | test | `validation-*.parquet`, `test-*.parquet` |
| `commonsenseqa` | train | validation | `train-*.parquet`, `validation-*.parquet` |
| `strategyqa` | train | test | `train-*.parquet`, `test-*.parquet` |

Parquet shards and filenames containing hashes are supported; `train.parquet`, `test.parquet`, and `validation.parquet` are also accepted. MATH subsets must be prepared in Hugging Face `Dataset.save_to_disk` format using the directory names above.

Keep the original dataset fields: `question` and `answer` for GSM8K; `problem` and `solution` for MATH; `question`, `options`, `answer`, and `cot_content` for MMLU-Pro; `question`, `choices`, and `answerKey` for CommonsenseQA; `question` and a Boolean `answer` for StrategyQA.

CommonsenseQA and StrategyQA require model-generated Self-CoT support responses, described below. Datasets, model weights, and historical experiment outputs are not included.

## Main Experiments

Run the following commands from the repository root. This example uses Qwen2.5-Math-7B on GSM8K.

```bash
COMMON=(
  --model qwen2.5-math-7b
  --dataset gsm8k
  --model-path /path/to/Qwen2.5-Math-7B
  --data-path /path/to/gsm8k
  --devices 0
)
```

### Extract and evaluate vectors

```bash
bash scripts/main_experiments/extract_vectors.sh "${COMMON[@]}"
bash scripts/main_experiments/eval_extracted_vectors.sh "${COMMON[@]}"
```

### Learn and evaluate vectors

```bash
bash scripts/main_experiments/learn_vectors.sh "${COMMON[@]}" --epochs 1
bash scripts/main_experiments/eval_learned_vectors.sh "${COMMON[@]}"
```

Each model and dataset has its own vectors. By default, the scripts sweep all layers and report the highest evaluation accuracy. Learning and evaluation run independently for each layer; every evaluation inserts a vector into exactly one layer. Extraction can collect vectors for all layers in one pass over the support set.

Use `--layers 3` for a single layer or `--layers 0,3,7` for a subset. Acquire and evaluate vectors using the same model, dataset, and layer indices.

```bash
bash scripts/main_experiments/learn_vectors.sh "${COMMON[@]}" --layers 3 --epochs 1
bash scripts/main_experiments/eval_learned_vectors.sh "${COMMON[@]}" --layers 3

# Inspect the command without loading a model.
bash scripts/main_experiments/learn_vectors.sh "${COMMON[@]}" --layers 3 --dry-run
```

### CommonsenseQA and StrategyQA

Existing Self-CoT files can be passed directly to the acquisition scripts. Use JSONL records with `question`, `self_cot`, and `is_correct`; `gt_answer` or `gt_numerical` can be included to check the ground-truth answer. JSONL content with a `.json` extension is also supported.

```json
{"question": "Example question", "gt_answer": "B", "self_cot": "Reasoning ... The answer is B.", "is_correct": true}
```

To generate Self-CoT responses from the support split:

```bash
python -m src.generate_self_cot \
  --model qwen2.5-math-7b \
  --dataset commonsenseqa \
  --model-path /path/to/Qwen2.5-Math-7B \
  --data-path /path/to/commonsenseqa \
  --output results/self_cot/commonsenseqa_qwen.jsonl \
  --devices 0
```

This saves all responses and a separate `_correct_only.jsonl` file. Vector acquisition uses correct records matched to support questions. Use Self-CoT responses from the same model and dataset.

```bash
COMMON=(
  --model qwen2.5-math-7b
  --dataset commonsenseqa
  --model-path /path/to/Qwen2.5-Math-7B
  --data-path /path/to/commonsenseqa
  --devices 0
)
SELF_COT=results/self_cot/commonsenseqa_qwen_correct_only.jsonl

bash scripts/main_experiments/extract_vectors.sh "${COMMON[@]}" --self-cot-path "$SELF_COT"
bash scripts/main_experiments/eval_extracted_vectors.sh "${COMMON[@]}"

bash scripts/main_experiments/learn_vectors.sh "${COMMON[@]}" --self-cot-path "$SELF_COT" --epochs 1
bash scripts/main_experiments/eval_learned_vectors.sh "${COMMON[@]}"
```

For StrategyQA, replace `--dataset`, `--data-path`, and the Self-CoT file with the corresponding values.

### Arguments

| Argument | Default / usage |
| --- | --- |
| `--layers` | `all`; also accepts a single index or comma-separated indices |
| `--support-samples` | Override the acquisition support size; `0` uses all available support |
| `--eval-samples` | `1000` |
| `--self-cot-path` | Required for CommonsenseQA / StrategyQA acquisition |
| `--epochs` | `1` |
| `--lr` | `5e-3` for Qwen layers 0–3; `1e-4` otherwise |
| `--warmup` | `0.1` for Qwen / GSM8K and all MMLU-Pro runs; `0.5` otherwise |
| `--batch-size` | Extraction: `1`; learning: `2`; evaluation: `16` |
| `--seed` | `42` |
| `--devices` | `0`; sets `CUDA_VISIBLE_DEVICES` |
| `--strategy` | `deepspeed_stage_2_offload` |
| `--output-dir` | `results/main_experiments/<model>/<dataset>/` |
| `--overwrite` | Replace existing target outputs |
| `--dry-run` | Print commands without running them |

Default acquisition support sizes:

| Method | GSM8K | MATH-Easy | MATH-Hard | MMLU-Pro | CommonsenseQA | StrategyQA |
| --- | --- | --- | --- | --- | --- | --- |
| Extraction | 3000 | 3000 | 3000 | 70 | 500 | 500 |
| Learning / Qwen | 3000 | 2000 | 2000 | 70 | All correct Self-CoT | All correct Self-CoT |
| Learning / Llama | 2000 | 1000 | 1000 | 70 | All correct Self-CoT | All correct Self-CoT |

If fewer support examples are available, all available examples are used. Override the defaults with `--support-samples`, `--epochs`, `--lr`, and `--warmup` as needed. Use a separate output directory when changing the model, dataset, or hyperparameters.

## Outputs

```text
results/main_experiments/<model>/<dataset>/
├── extracted/
│   ├── vectors.pt
│   └── vectors.json
├── learned/layer-<L>/
│   ├── manifest.json
│   └── ckpt/.../epoch-<E>/encoder.pth
└── evaluation/
    ├── extracted/
    │   ├── layer-<L>.json
    │   └── summary.json
    └── learned/
        ├── layer-<L>.json
        └── summary.json
```

`summary.json` contains per-layer accuracy, the best layer, and whether the sweep covered every layer. A summary is written only when all requested results are present and have the expected evaluation sample count.

To summarize existing results:

```bash
bash scripts/main_experiments/eval_learned_vectors.sh "${COMMON[@]}" --summarize-only
```

## Code Structure

| Path | Description |
| --- | --- |
| `scripts/main_experiments/` | Four Bash entry points for acquisition and evaluation |
| `scripts/run_experiments.py` | Argument handling, layer sweeps, output tracking, and summaries |
| `src/extract_cot_vector.py` | Vector extraction |
| `src/train.py`, `src/shift_model.py` | Vector learning |
| `src/eval.py`, `src/shift_encoder.py` | Vector loading, hooks, and insertion |
| `src/generate_self_cot.py` | Support-only Self-CoT generation |
| `src/dataset_utils/` | Data loading, prompts, answer labels, and evaluation |
| `src/config/` | Extraction, learning, and evaluation configurations |
| `testbed/` | Shared model interfaces and hook utilities |
| `tests/` | Launcher, configuration, data pipeline, and answer-label tests |

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover configuration composition, data processing, launch commands, and answer-only CE label handling. GPU end-to-end runs and paper scores have not been verified in this release. Data selection, token construction, and answer parsing have been refactored; numerical equivalence with historical experiments has not been established.

## Citation

```bibtex
@inproceedings{li2026cot,
  title={Cot vectors: Transferring and probing the reasoning mechanisms of llms},
  author={Li, Li and Wang, Ziyi and Wu, Yongliang and Cai, Jianfei and others},
  booktitle={International Conference on Learning Representations},
  volume={2026},
  pages={108745--108767},
  year={2026}
}
```

## License

[MIT](LICENSE)
