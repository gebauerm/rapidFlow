import os
import shutil
from rapidflow.experiments.experiment import Experiment
from sklearn.naive_bayes import GaussianNB
from rapidflow.metrics_handler import MetricsHandler
from rapidflow.objective import Objective


class MockObjective(Objective):
    def __init__(self, train_data, val_data, test_data) -> None:
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def train(self, trial):
        hyperparameters = dict(var_smoothing=trial.suggest_float("var_smoothing", 1e-9, 1e-5, log=True))
        model = GaussianNB(**hyperparameters)
        self.track_model(model, hyperparameters)

        self.model.fit(self.train_data[0], y=self.train_data[1])
        preds = self.model.predict(self.val_data[0])
        metrics_handler = MetricsHandler()
        classification_metrics = metrics_handler.calculate_classification_metrics(self.val_data[1], preds)
        return classification_metrics["macro avg f1-score"]

    def test(self):
        preds = self.model.predict(self.test_data[0])
        metrics_handler = MetricsHandler()
        classification_metrics = metrics_handler.calculate_classification_metrics(self.test_data[1], preds)
        return classification_metrics


def test_experiment_cpu(generate_test_data):
    # set up
    exp_path = os.path.abspath(os.path.dirname(__file__))
    objective_cls = MockObjective
    train_data, val_data, test_data = generate_test_data
    experiment = Experiment(experiment_path=exp_path, title="test_exp")
    result_path = os.path.join(exp_path, "exp__test_exp__")

    # perform
    experiment.add_objective(objective_cls=objective_cls, args=[train_data, val_data, test_data])
    experiment.run(2, 2, 2)

    # assert
    assert os.path.isdir(result_path)
    assert os.path.isfile(os.path.join(result_path, "averaged_test_results.json"))

    # cleanup
    shutil.rmtree(result_path)


def test_failing():
    assert 1 == 0


if __name__ == "__main__":
    import pytest as pt
    pt.main()
