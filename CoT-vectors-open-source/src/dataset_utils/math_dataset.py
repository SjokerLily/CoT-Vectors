"""MATH-Easy/Hard: precomputed Hugging Face save_to_disk splits."""
from pathlib import Path
from .answers import math_answer
from .common import ReasoningDataset


class MATHDataset(ReasoningDataset):
    support_datasets = ["math_dataset"]
    question_key = "problem"
    answer_key = cot_key = "solution"
    eval_system_prompt = r"You are a helpful and precise assistant for solving math problems. Please reason step by step, and put your final answer within \boxed{}."
    teacher_system_prompt = eval_system_prompt
    student_system_prompt = "You are a helpful and precise assistant for solving math problems. Output the final numerical answer."
    extract_answer = staticmethod(math_answer)

    def _load_splits(self, local_path):
        from datasets import Dataset

        subset = getattr(self.cfg, "subset", None)
        if subset not in ("easy", "hard"):
            raise ValueError("The main MATH experiments require subset=easy or subset=hard")
        self.subset_type = subset
        paths = [Path(local_path) / f"math_{split}_subset_{subset}" for split in ("train", "test")]
        for path in paths:
            if not path.is_dir():
                raise FileNotFoundError(f"Prepared MATH split not found: {path}")
        return tuple(Dataset.load_from_disk(str(path)) for path in paths)
