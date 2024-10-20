import os

from joblib import load
from sklearn.naive_bayes import GaussianNB
import torch

from rapidflow.experiments.model_saver import ModelSaver
from rapidflow.nn_model import NNModel


def test_model_saver():
    # set up
    exp_path = os.path.abspath(os.path.dirname(__file__))
    torch_model_save_path = os.path.join(exp_path, "PytestTorchModel.pt")
    sklearn_model_save_path = os.path.join(exp_path, "PytestSKLearnModel.gzip")

    torch_model = NNModel(lr=0.005, weight_decay=None)
    sklearn_model = GaussianNB()

    torch_saver = ModelSaver()
    sklearn_saver = ModelSaver()

    torch_saver.set_strategy(torch_model)
    sklearn_saver.set_strategy(sklearn_model)

    # perform
    torch_saver.save_model(torch_model, "PytestTorchModel", exp_path)
    sklearn_saver.save_model(sklearn_model, "PytestSKLearnModel", exp_path)

    loaded_torch_model = NNModel(lr=0.005, weight_decay=None)
    loaded_torch_model.load_state_dict(torch.load(torch_model_save_path))
    loaded_torch_model.eval()
    loaded_sklearn_model = load(sklearn_model_save_path)

    torch_equality = True
    for p1, p2 in zip(torch_model.parameters(), loaded_torch_model.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            torch_equality = False

    # assert
    assert torch_equality
    assert sklearn_model.get_params() == loaded_sklearn_model.get_params()

    # cleanup
    os.remove(torch_model_save_path)
    os.remove(sklearn_model_save_path)


if __name__ == "__main__":
    import pytest as pt
    pt.main()
