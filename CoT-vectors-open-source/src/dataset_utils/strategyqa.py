"""StrategyQA: training Self-CoT support and labeled test queries."""
from .answers import boolean_answer
from .common import ReasoningDataset


class StrategyQADataset(ReasoningDataset):
    support_datasets = ["strategyqa"]
    requires_self_cot = True
    eval_system_prompt = r"You are a helpful and precise assistant for solving problems. Please reason step by step, and put your final answer within \boxed{}.Your final output should be only True or False."
    teacher_system_prompt = student_system_prompt = eval_system_prompt
    extract_answer = staticmethod(boolean_answer)
