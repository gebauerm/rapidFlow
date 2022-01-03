from abc import abstractmethod
import torch.nn as nn
from typing import Any


class OptimizerCallClass(type):
    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        """
        Metaclass Magic, have to revisit how I did that.

        Returns:
            Any: [description]
        """
        instance = super().__call__(*args, **kwargs)
        instance.create_optimizer()
        return instance


class NNModel(nn.Module, metaclass=OptimizerCallClass):

    def __init__(self, lr, weight_decay):
        super().__init__()
        self.device = None
        self.lr = lr
        self.weight_decay = weight_decay

    @abstractmethod
    def loss_function(self, data, *args, **kwargs):
        pass

    @abstractmethod
    def forward(self, x, *args, **kwargs):
        pass

    @abstractmethod
    def create_optimizer(self):
        pass

    @abstractmethod
    def train_step(self, data, *args, **kwargs):
        pass

    @abstractmethod
    def test_step(self, data, *args, **kwargs):
        pass
