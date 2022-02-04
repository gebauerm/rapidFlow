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
    """
    An Experiment Class provides multi-process execution and hyperparameter optimization, if needed.
    It is build upon optuna and simplifies some things that are expected by optuna.
    An experiment creates a folder and runs your objective on a test set.

    The experiment is performed k-times as metrics from the model tasks
    (accuracy, f1-measure etc.) are random numbers and need basic confidence intervals, which are
    currently provided by the standard deviation of every k run.
    A trial sets the number of runs per hyperparameter setting that is randomly generated.
    Meaning that if:
        k = 2
        trials = 50
    there will be trained and evaluated 100 models on the train and validation set. Afterwards the model
    which performed the best on the validation set is saved and evaluated on the test set.

    The number of processes sets the number of CPU cores used. It is expected that CUDA manages GPU
    workload distribution per CPU core. Thus one GPU per CPU-core as nothing other is defined in the
    objective.
    """
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
        self.objective_args = None
        self.objective_cls = None

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

    def _run(self, idx):
        """Starts an experiment study. An experiment study manages parallelization. The best model is
        evaluated herin and saved as its results as well.

        Args:
            idx ([type]): [description]
            trials ([type]): [description]
            num_processes ([type]): [description]

        Returns:
            [type]: [description]
        """
        best_trial = self.experiment_study.run()
        test_result, objective = self.evaluate(best_trial)
        self.study_result_saver.save_study_results(idx, test_result, objective)
        return test_result

    def run(self, k, trials, num_processes=None):
        """Starts the experiment. and serves as a wrapper. After each finished run the best model is evaluated
        on the test set.

        Args:
            k ([type]): number of repitions of trials
            trials ([type]): number of randomly generated hyperparameter settings used per training
            num_processes ([type], optional): [description]. Used CPU cores
        """
        logger.info(20*"-" + " Starting Experiment!" + 20*"-")
        start = timer()
        test_results = []
        self.experiment_study = ExperimentStudy(
            self.title, self.objective_cls, self.objective_args, trials, num_processes, self.experiment_path)
        for idx in range(k):
            logger.info(f"Starting run for k={idx+1}")
            test_result = self._run(idx)
            test_results.append(test_result)
        averaged_test_results = self.metrics_handler.average_results(test_results)
        self.study_result_saver.save_study_metrics(
            averaged_test_results, "averaged_test_results", self.experiment_path)
        elapsed_time = timedelta(seconds=timer()-start)
        logger.info(20*"-" + f" Experiment finished - Elapsed Time: {elapsed_time} " + 20*"-")

    def evaluate(self, best_trial):
        """
        Evalutes the best performing model from the validation set on the test set.

        Args:
            best_trial ([type]): [description]

        Returns:
            [type]: [description]
        """
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
