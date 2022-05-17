import json
import shutil
import os

from rapidflow.experiments.model_saver import ModelSaver


class StudyResultSaver:

    def __init__(self, experiment_path) -> None:
        self.experiment_path = experiment_path
        self.model_saver = ModelSaver()

    def save_best_model(self, idx, model, hyperparameters, path):
        """
        Saves the best model of a study of the experiment class. Currently it only saves torch and sklearn models.

        Args:
            idx ([type]): [description]
            model ([type]): [description]
            hyperparameters ([type]): [description]
            path ([type]): [description]
        """
        self.model_saver.set_strategy(model)
        self.model_saver.save_model(model, f"best_model_{idx}", path)
        self.save_hyperparameters(hyperparameters, f"best_model_parameters_{idx}", path)

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
