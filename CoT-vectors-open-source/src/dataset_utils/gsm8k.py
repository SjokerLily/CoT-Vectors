"""GSM8K: official train support, test queries and ground-truth CoT."""
from .answers import numeric_answer
from .common import ReasoningDataset


class GSM8KDataset(ReasoningDataset):
    support_datasets = ["gsm8k"]
    cot_key = "answer"
    eval_system_prompt = r"Answer the question thinking step by step, and put your final answer within \boxed{}."
    teacher_system_prompt = eval_system_prompt
    student_system_prompt = r"Answer the question immediately, without any elaboration, and put your final answer within \boxed{}."
    extract_answer = staticmethod(numeric_answer)
