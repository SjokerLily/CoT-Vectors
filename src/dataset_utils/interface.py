"""Lightweight dataset interface; importing adapters does not load a model."""
from abc import ABC, abstractmethod


class DatasetBase(ABC):
    support_datasets = []

    def __init__(self, data_cfg, model_processor=None, model_name=None):
        self.cfg = self.data_cfg = data_cfg
        self.model_processor = model_processor
        self.model_name = model_name
        if self.name not in self.support_datasets:
            raise ValueError(f"{self.name} is not supported by {type(self).__name__}")

    @property
    def name(self):
        return self.cfg.name

    @staticmethod
    def metric_key():
        return "accuracy"

    @property
    def instruction(self):
        return ""

    @property
    def num_role_in_round(self):
        return 2

    @property
    def support_set(self):
        return self._support_set

    @property
    def query_set(self):
        return self._query_set

    @abstractmethod
    def extract_answer(self, prediction):
        raise NotImplementedError

    @abstractmethod
    def eval(self, eval_cfg, model):
        raise NotImplementedError

    @abstractmethod
    def train_dataloader(self, model, batch_size, distributed=True):
        raise NotImplementedError

    @abstractmethod
    def validation_dataloader(self, batch_size):
        raise NotImplementedError
