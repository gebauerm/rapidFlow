import os
import shutil
from datetime import timedelta
from timeit import default_timer as timer
from rapidflow.metrics_handler import MetricsHandler
from rapidflow.experiments.experiment_study import ExperimentStudy
from rapidflow.experiments.study_result_saver import StudyResultSaver
from torch.multiprocessing import set_start_method
from rapidflow.utils.experiment_logger import logger


class Experiment:
    def __init__(
            self, title=None, experiment_path=None, model_name=None,) -> None:
        if title:
            self.title = title
        else:
            self.title = "no title"

        if experiment_path:
            self.experiment_path = self.create_experiment_folder(
                experiment_path, model_name=model_name)

        self.metrics_handler = MetricsHandler()
        self.study_result_saver = StudyResultSaver(self.experiment_path)
        self.objective_func = None
        self.objective_args = None

    def create_experiment_folder(
            self, experiment_file_path, model_name, overwrite=True):
        # dict_byte_string = json.dumps(hyperparameter).encode("utf-8")
        # = hashlib.sha1(dict_byte_string).hexdigest()
        if not model_name:
            model_name = ""
        folder_name = f"exp__{self.title}__{model_name}"
        experiment_path = os.path.join(experiment_file_path, folder_name)
        if os.path.exists(experiment_path) and overwrite:
            shutil.rmtree(experiment_path)
        os.mkdir(experiment_path)
        return experiment_path

    def _run(self, idx, trials, num_processes):

        self.experiment_study = ExperimentStudy(
            self.title, self.objective_cls, self.objective_args, trials, num_processes, self.experiment_path)
        best_trial = self.experiment_study.run()
        test_result, objective = self.evaluate(best_trial)
        self.study_result_saver.save_study_results(idx, test_result, objective)
        return test_result

    def run(self, k, trials, num_processes=None):
        logger.info(20*"-" + " Starting Experiment!" + 20*"-")
        set_start_method("spawn")
        start = timer()
        test_results = []
        for idx in range(k):
            logger.info(f"Starting run for k={idx}")
            test_result = self._run(idx, trials, num_processes)
            test_results.append(test_result)
        averaged_test_results = self.metrics_handler.average_results(test_results)
        self.study_result_saver.save_study_metrics(
            averaged_test_results, "averaged_test_results", self.experiment_path)
        elapsed_time = timedelta(seconds=timer()-start)
        logger.info(20*"-" + f" Experiment finished - Elapsed Time: {elapsed_time} " + 20*"-")

    def evaluate(self, best_trial):
        if self.objective_cls:
            logger.info("Starting Evaluation...")
            objective = self.objective_cls(*self.objective_args)
            objective(best_trial)
            test_result = objective.test()
        else:
            test_result = None
        return test_result, objective

    def add_objective(self, objective_cls, args):
        self.objective_cls = objective_cls
        self.objective_args = args
