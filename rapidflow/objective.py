from abc import ABC, abstractmethod
import torch
import numpy
import random


class Objective(ABC):
    """An Objective serves as a BaseClass for any objective function that is going to be used within the
    framework. The execution content is to some degree arbitrary, as you need to make sure to have a
    model with hyperparameters.
    The Objective is optimized within the Optuna Framework. The training/optimization of the is started
    by calling the __call__ method of the objective class.

    Args:
        ABC ([type]): [description]
    """

    def __init__(self) -> None:
        self._hyperparameters = None
        self._model = None

    def __call__(self, trial):
        torch.manual_seed(1337)
        numpy.random.seed(1337)
        random.seed(1337)
        target_val = self.train(trial)
        return target_val

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def train(self, trial):
        pass

    @property
    def model(self):
        if self._model:
            return self._model
        else:
            raise AttributeError("Model not tracked! Please make sure to call 'self.track_model(model, hyperparameters)' in your Objective Class!")

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def hyperparameters(self):
        if self._hyperparameters:
            return self._hyperparameters
        else:
            raise AttributeError("Model not tracked! Please make sure to call 'self.track_model(model, hyperparameters)' in your Objective Class!")

    @hyperparameters.setter
    def hyperparameters(self, hyperparameters):
        self._hyperparameters = hyperparameters

    def track_model(self, model, hyperparameters):
        self.model = model
        self.hyperparameters = hyperparameters
