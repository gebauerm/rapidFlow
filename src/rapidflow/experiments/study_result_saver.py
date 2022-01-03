import torch
import json
import shutil
import os


class StudyResultSaver:

    def __init__(self, experiment_path) -> None:
        self.experiment_path = experiment_path

    def save_best_model(self, idx, model, hyperparameters, path):
        """
        Saves the best model of a study of the experiment class. Currently it only saves torch models.

        Args:
            idx ([type]): [description]
            model ([type]): [description]
            hyperparameters ([type]): [description]
            path ([type]): [description]
        """
        try:
            self.save_torch_model(model, f"best_model_{idx}", path)
        except AttributeError:
            print(Warning("Your model is not a torch model. Currently saving for non torch models is not supported!"))
        self.save_hyperparameters(hyperparameters, f"best_model_parameters_{idx}", path)

    def save_torch_model(self, model, title, path):
        torch.save(model.state_dict(), os.path.join(path, f"{title}.pt"))

    def save_hyperparameters(self, hyperparameters, title, path):
        with open(os.path.join(path, f"{title}.json"), "w") as f:
            json.dump(hyperparameters, f)

    def save_study_metrics(self, test_results, title, path):
        with open(os.path.join(path, f"{title}.json"), "w") as f:
            json.dump(test_results, f)

    def save_study_results(self, idx, study_results, objective, overwrite=True):
        study_result_path = os.path.join(self.experiment_path, f"study_{idx}")
        if os.path.exists(study_result_path) and overwrite:
            shutil.rmtree(study_result_path)
        os.mkdir(study_result_path)
        self.save_best_model(idx, objective.model, objective.hyperparameters, study_result_path)
        self.save_study_metrics(study_results, f"test_results_{idx}", study_result_path)
