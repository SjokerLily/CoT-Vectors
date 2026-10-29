"""CommonsenseQA: training Self-CoT support, labeled validation queries."""
from .answers import choice_answer
from .common import ReasoningDataset, formatted_choices, parquet_splits


class CommonsenseQADataset(ReasoningDataset):
    support_datasets = ["commonsenseqa"]
    answer_key = "answerKey"
    requires_self_cot = True
    eval_system_prompt = r"You are a helpful and precise assistant for solving problems. Please reason step by step, and put your final answer within \boxed{}.Your final output should be only the uppercase letter of the correct choice (e.g., A)."
    teacher_system_prompt = r"You are a helpful and precise assistant for solving math problems. Please reason step by step, and put your final answer within \boxed{}.Your final output should be only the uppercase letter of the correct choice (e.g., A)."
    student_system_prompt = r"You are a helpful and precise assistant for solving problems. Please directly put your final answer within \boxed{}.Your final output should be only the uppercase letter of the correct choice (e.g., A)."

    @staticmethod
    def extract_answer(prediction):
        return choice_answer(prediction, choices="ABCDE")

    def _load_splits(self, local_path):
        return parquet_splits(local_path, support="train", query="validation")

    def question(self, item):
        return str(item["question"]) + formatted_choices(item["choices"])
