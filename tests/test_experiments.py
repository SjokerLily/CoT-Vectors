"""CPU-only contract tests for the main experiments launcher (no model imports)."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("run_experiments", Path(__file__).resolve().parents[1] / "scripts/run_experiments.py")
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)

class ExperimentTests(unittest.TestCase):
    def test_layer_selection(self):
        self.assertEqual(t.layers_from("all", 32), list(range(32)))
        self.assertEqual(t.layers_from("3", 28), [3])
        for value in ("28", "-1", "1,1", "x"):
            with self.assertRaises(ValueError):
                t.layers_from(value, 28)

    def test_commands_for_all_model_dataset_actions(self):
        for model in t.MODELS:
            for dataset in t.TASKS:
                for action in ("extract", "learn", "eval-extracted", "eval-learned"):
                    with self.subTest(model=model, dataset=dataset, action=action):
                        with patch("builtins.print") as output, patch.object(t.subprocess, "run") as run:
                            t.main([action, "--model", model, "--dataset", dataset,
                                    "--model-path", "/model with spaces", "--data-path", "/data with spaces",
                                    "--layers", "3", *(["--self-cot-path", "/support.jsonl"] if dataset in t.SELF_COT_TASKS else []), "--dry-run"])
                            run.assert_not_called()
                            command = output.call_args.args[0]
                            self.assertIn("--config-name", command)
                            if action != "extract":
                                self.assertIn("only_shift_at_layer=3", command)
                            if dataset in t.SELF_COT_TASKS and action in ("extract", "learn"):
                                self.assertIn("SELF_COT", command)
                            else:
                                self.assertNotIn("SELF_COT", command)

    def fixture(self, root):
        model = root / "model"; model.mkdir()
        t.write_json(model / "config.json", {"num_hidden_layers": 28})
        data = root / "data"; data.mkdir()
        return ["--model", "qwen2.5-math-7b", "--dataset", "gsm8k",
                "--model-path", str(model), "--data-path", str(data),
                "--output-dir", str(root / "outputs"), "--layers", "0,3"]

    def test_acquisition_checkpoint_roundtrip_and_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); args = self.fixture(root); calls=[]
            def fake_run(command, **kwargs):
                values={v.partition("=")[0]: v.partition("=")[2] for v in command if "=" in v}
                calls.append((values, dict(kwargs["env"])))
                if command[1].endswith("train.py"):
                    target=Path(kwargs["env"]["COT_RESULTS_DIR"]) / "ckpt/main_experiments-qwen2.5-math-7b-gsm8k-3000/epoch-0/encoder.pth"
                    target.parent.mkdir(parents=True); target.touch()
                else:
                    target=Path(json.loads(values["record_path"]))
                    layer=int(values["only_shift_at_layer"])
                    t.write_json(target, {"records":[{}]*2, "eval_result":{"accuracy":0.5 if layer==0 else 1.0}})
            with patch.object(t.subprocess,"run",side_effect=fake_run), patch("builtins.print"):
                t.main(["learn", *args])
                t.main(["eval-learned", *args, "--eval-samples", "2"])
            self.assertEqual([int(v["only_shift_at_layer"]) for v,e in calls], [0,3,0,3])
            self.assertIn("epoch-0", calls[-1][0]["ckpt_path"])
            summary=t.read_json(root / "outputs/evaluation/learned/summary.json")
            self.assertEqual(summary["best_layer"],3)
            self.assertFalse(summary["full_layer_scan"])
            with patch.object(t.subprocess,"run") as run, self.assertRaises(FileExistsError), patch("builtins.print"):
                t.main(["learn", *args])
            run.assert_not_called()

    def test_failure_propagates(self):
        with tempfile.TemporaryDirectory() as folder:
            args=self.fixture(Path(folder))
            with patch.object(t.subprocess,"run",side_effect=subprocess.CalledProcessError(1,["python"])), patch("builtins.print"):
                with self.assertRaises(subprocess.CalledProcessError):
                    t.main(["learn",*args])

    def test_summary_rejects_partial_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)
            t.write_json(directory / "layer-0.json",{"records":[{}],"eval_result":{"accuracy":1.0}})
            with self.assertRaises(ValueError): t.summarize(directory,[0],28,1000)
            self.assertFalse((directory / "summary.json").exists())

    def test_full_scan_commands_are_separate_single_layer_runs(self):
        for model,count in t.MODELS.items():
            with patch('builtins.print') as output:
                t.main(['learn','--model',model,'--dataset','math-hard',
                        '--model-path','/model','--data-path','/data','--dry-run'])
            self.assertEqual(output.call_count,count)
            for layer,call in enumerate(output.call_args_list):
                self.assertIn('only_shift_at_layer='+str(layer),call.args[0])

    def test_extract_layer_and_identity_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);args=self.fixture(root)
            identity={"model":"qwen2.5-math-7b","dataset":"gsm8k","model_path":str(root/'model'),"data_path":str(root/'data'),"layers":[0]}
            t.write_json(root/'outputs/extracted/vectors.json',identity)
            with patch.object(t.subprocess,"run"), patch("builtins.print"), self.assertRaises(ValueError):
                t.main(["eval-extracted",*args,"--layers","3"])
            identity["dataset"]="mmlu-pro"
            t.write_json(root/'outputs/extracted/vectors.json',identity)
            with patch.object(t.subprocess,"run"), patch("builtins.print"), self.assertRaises(ValueError):
                t.main(["eval-extracted",*args,"--layers","0"])

if __name__ == "__main__": unittest.main()
