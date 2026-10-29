"""Adapters for all datasets in the final paper's main experiments."""
from .interface import DatasetBase
from .gsm8k import GSM8KDataset
from .math_dataset import MATHDataset
from .mmlu import MMLUProDataset
from .commonsenseqa import CommonsenseQADataset
from .strategyqa import StrategyQADataset

dataset_mapping = {
    "gsm8k": GSM8KDataset,
    "math_dataset": MATHDataset,
    "mmlu-pro": MMLUProDataset,
    "commonsenseqa": CommonsenseQADataset,
    "strategyqa": StrategyQADataset,
}

__all__ = ["dataset_mapping", "DatasetBase"]
