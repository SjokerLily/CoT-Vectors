"""Exercise the actual training forward with lightweight stand-ins for model calls."""
import ast
from contextlib import nullcontext
import enum
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

class Tensor(np.ndarray):
    def __new__(cls, values):
        return np.asarray(values).view(cls)
    def bool(self):
        return self.astype(bool)
    def masked_fill(self, mask, value):
        result = self.copy()
        result[np.asarray(mask, dtype=bool)] = value
        return result

def training_forward():
    path = Path(__file__).resolve().parents[1] / "src/shift_model.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    strategy = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "Strategy")
    model = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "ShiftModel")
    forward = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    namespace = {"enum": enum, "torch": SimpleNamespace(bool=np.dtype(bool), no_grad=nullcontext)}
    exec(compile(ast.Module(body=[strategy, forward], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["forward"], namespace["Strategy"]

class TrainingLabelTests(unittest.TestCase):
    def run_forward(self, labels):
        forward, strategy = training_forward()
        ids = Tensor([[0, 2, 12, 13, 0], [0, 0, 3, 21, 0]])
        attention = Tensor([[0, 1, 1, 1, 1], [0, 0, 1, 1, 1]])
        received, kl_masks = [], []
        class Model:
            disable_adapter = staticmethod(nullcontext)
            def __call__(self, **kwargs):
                if "labels" in kwargs:
                    received.append(kwargs["labels"].copy())
                return {"logits": None, "loss": 2.0}
        module = SimpleNamespace(
            lmm=SimpleNamespace(model=Model(), processor=SimpleNamespace(
                pad_token="<pad>", pad_token_id=0, bos_token_id=1, eos_token="<eos>")),
            shift_encoder=SimpleNamespace(register_record_hooks=lambda: {}, register_shift_hooks=lambda: {}),
            strategy=strategy.LM_LOSS | strategy.LOGITS_KL_DIV,
            is_lora=False,
            cfg=SimpleNamespace(dft_loss=False, ce_loss_weight=.5, align_loss_weight=1.0),
            remove_hooks=lambda hooks: None,
        )
        def kl(student, teacher, student_mask, teacher_mask):
            kl_masks.append(student_mask.copy())
            return {"logits_kl_loss": 0.0}
        module.calculate_logits_kl_loss = kl
        original = labels.copy()
        output = forward(module, prefix_inputs={"input_ids": ids, "attention_mask": attention},
                         teacher_answer_mask=Tensor([[False, False, True, True, False],
                                                    [False, False, False, True, False]]),
                         teacher_cot_mask=None, student_inputs={"input_ids": ids, "attention_mask": attention},
                         student_labels=labels, images=[[], []])
        np.testing.assert_array_equal(labels, original)
        self.assertEqual(output["ce_loss"], 2.0)
        self.assertEqual(output["loss"], 1.0)
        self.assertEqual(len(received), 1)
        expected_kl_mask = labels.bool() if labels.dtype == np.dtype(bool) else (labels != -100) & attention.bool()
        np.testing.assert_array_equal(kl_masks[0], expected_kl_mask)
        return received[0]

    def test_integer_answer_labels_reach_language_model(self):
        labels = Tensor([[-100, -100, 12, 13, -100], [-100, -100, -100, 21, -100]])
        np.testing.assert_array_equal(self.run_forward(labels), labels)

    def test_legacy_boolean_answer_mask_is_supported(self):
        labels = Tensor([[False, False, True, True, False], [False, False, False, True, False]])
        expected = [[-100, -100, 12, 13, -100], [-100, -100, -100, 21, -100]]
        np.testing.assert_array_equal(self.run_forward(labels), expected)

    def test_padding_is_ignored_even_if_label_is_present(self):
        labels = Tensor([[99, -100, 12, 13, -100], [98, 99, -100, 21, -100]])
        expected = [[-100, -100, 12, 13, -100], [-100, -100, -100, 21, -100]]
        np.testing.assert_array_equal(self.run_forward(labels), expected)

if __name__ == "__main__":
    unittest.main()
