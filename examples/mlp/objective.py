import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
from rapidflow.objective import Objective
from examples.mlp.mlp import MLP
from rapidflow.metrics_handler import MetricsHandler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from rapidflow.experiments.experiment import Experiment


class MLPObjective(Objective):

    def __init__(self, train_loader, val_loader, test_loader) -> None:
        super().__init__()
        self.train_loader = train_loader
        self.val_laoder = val_loader
        self.test_loader = test_loader

    def train(self, trial=None):
        # for optuna usage
        hyperparameters = dict(
            input_size=20,
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True),
            hidden_layer_config=[
                trial.suggest_int(f"layer_{idx}_size", 10, 100)
                for idx in range(trial.suggest_int("n_layers", 1, 3))],
            output_size=1
        )

        metrics_handler = MetricsHandler()
        # model setup
        model = MLP(**hyperparameters)
        self.track_model(model, hyperparameters)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        # train
        for epoch in tqdm.tqdm(range(1, 101)):
            for x, y in self.train_loader:
                x = x.to(device)
                y = y.to(device)
                self.model.train()
                _ = self.model.train_step(x, y)
        # validation
        val_targets = []
        val_preds = []
        for x, y in self.val_laoder:
            x = x.to(device)
            y = y.to(device)
            predictions = self.model.test_step(x)
            targets = y.flatten().detach()
            val_targets += [targets.detach()]
            val_preds += [predictions.detach()]
        val_targets = torch.cat(val_targets).cpu().numpy()
        val_preds = torch.cat(val_preds).cpu().numpy()
        classification_metrics = metrics_handler.calculate_classification_metrics(
                    val_targets, val_preds)
        return classification_metrics['macro avg f1-score']

    def test(self):
        metrics_handler = MetricsHandler()
        model = self.model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        test_targets = []
        test_predictions = []
        for x, y in self.test_loader:
            x = x.to(device)
            y = y.to(device)
            predictions = model.test_step(x)
            targets = y.flatten().detach()
            test_targets += [targets.detach()]
            test_predictions += [predictions.detach()]
        test_targets = torch.cat(test_targets).cpu().numpy()
        test_predictions = torch.cat(test_predictions).cpu().numpy()
        test_metrics = metrics_handler.calculate_classification_metrics(
            test_targets, test_predictions)
        return test_metrics


if __name__ == "__main__":
    # random dataset
    x, y = make_classification(n_samples=2000, n_features=20, n_classes=2)
    x_train, x_test, y_train, y_test = train_test_split(
        torch.from_numpy(x).to(torch.float32), torch.from_numpy(y).to(torch.float32), test_size=.2)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=.2)
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=50)
    val_loader = DataLoader(val_dataset, batch_size=50)
    test_loader = DataLoader(test_dataset, batch_size=50)
    # setting up an experiment
    experiment = Experiment(
        experiment_path=os.path.abspath(os.path.dirname(__file__)))
    experiment.add_objective(MLPObjective, args=[train_loader, val_loader, test_loader])
    experiment.run(k=2, trials=2, num_processes=1)
