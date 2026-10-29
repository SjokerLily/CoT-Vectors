"""MMLU-Pro: annotated validation support and disjoint test queries."""
from .answers import choice_answer
from .common import ReasoningDataset, formatted_choices, parquet_splits


class MMLUProDataset(ReasoningDataset):
    support_datasets = ["mmlu-pro"]
    eval_system_prompt = r"You are a helpful and precise assistant for solving problems. Please reason step by step, and put your final answer within \boxed{}.Your final output should be only the uppercase letter of the correct choice (e.g., A)."
    teacher_system_prompt = r"You are a helpful and precise assistant for solving math problems. Please reason step by step, and put your final answer within \boxed{}.Your final output should be only the uppercase letter of the correct choice (e.g., A)."
    student_system_prompt = "The following are multiple choice questions (with answers) about {$}. Do not think and directly output your final answer (X) where X is the correct letter choice."
    extract_answer = staticmethod(choice_answer)

    def _load_splits(self, local_path):
        return parquet_splits(local_path, support="validation", query="test")

    def question(self, item):
        return str(item["question"]) + formatted_choices(item["options"])
