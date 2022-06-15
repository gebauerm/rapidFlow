import os
from abc import ABC, abstractmethod
import torch
import sklearn
from joblib import dump


class ModelSaver:
    def __init__(self):
        self.strategies = {
            torch.nn.Module: TorchStrategy,
            sklearn.base.BaseEstimator: SKLearnStrategy
        }
        self.strategy = None

    def set_strategy(self, model):
        try:
            self.strategy = [value for key, value in self.strategies.items() if issubclass(type(model), key)][0]
        except IndexError:
            print(Warning(f"Your model is of type {type(model)}."
                          f"Currently saving is only supported for torch and sklearn models!"))

    def save_model(self, model, title, path):
        try:
            self.strategy.execute(model, title, path)
        except AttributeError:
            print(Warning("You have not set a strategy! Please call set_strategy() first!"))


class SavingStrategy(ABC):
    @abstractmethod
    def execute(model, title, path):
        pass


class TorchStrategy(SavingStrategy):
    @staticmethod
    def execute(model, title, path):
        torch.save(model.state_dict(), os.path.join(path, f"{title}.pt"))


class SKLearnStrategy(SavingStrategy):
    @staticmethod
    def execute(model, title, path):
        dump(model, os.path.join(path, f"{title}.gzip"))
