import unittest
from pathlib import Path
from unittest.mock import patch
import shlex
from test_experiments import t
try:
    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf
    HAS_HYDRA = True
except ImportError:
    HAS_HYDRA = False

@unittest.skipUnless(HAS_HYDRA, 'Hydra is required for config composition')
class ConfigTests(unittest.TestCase):
    def test_all_model_task_stage_configs(self):
        for model in t.MODELS:
            for task in t.TASKS:
                for action in ['extract','learn','eval-extracted','eval-learned']:
                    with self.subTest(model=model, task=task, action=action):
                        with patch('builtins.print') as output:
                            t.main([action,'--model',model,'--dataset',task,
                                    '--model-path','/model with spaces','--data-path','/data with spaces',
                                    '--layers','3',*(['--self-cot-path','/support.jsonl'] if task in t.SELF_COT_TASKS else []),'--dry-run'])
                        command=shlex.split(output.call_args.args[0])
                        with initialize_config_dir(config_dir=str(t.ROOT/'src/config'),version_base=None):
                            cfg=compose(config_name=command[3],overrides=command[4:])
                            OmegaConf.to_container(cfg,resolve=True)
                            if action!='extract':
                                self.assertEqual(cfg.encoder.cls._target_,'src.shift_encoder.AttnFFNShift')
                            self.assertEqual(cfg.data.name,t.TASKS[task][0])
                            self.assertEqual(cfg.data.get('use_self_cot',False), task in t.SELF_COT_TASKS and action in ('extract','learn'))
