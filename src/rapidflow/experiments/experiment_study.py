import optuna
import docker
import time
import torch
from multiprocessing import current_process, Queue, Process
from rapidflow.utils.experiment_logger import logger


class ExperimentStudy:

    def __init__(self, study_name, objective_cls, objective_args, trials, num_processes, experiment_path) -> None:
        self.storage_string = "postgresql://root:1234@localhost:5432/optuna_study"
        self.study_name = study_name
        self.objective_cls = objective_cls
        self.objective_args = objective_args
        self.trials = trials
        self.num_processes = num_processes
        self.multiprocessing = True if num_processes != 1 and num_processes is not None else False
        self.available_gpu_idxs = [i for i in range(torch.cuda.device_count())]  # TODO: GPU management necessary?

    def _start_postgres_db(self):
        if current_process().name == "MainProcess":
            self.client = docker.from_env()
            self.client.containers.run(
                "postgres:14.1", detach=True,
                environment=["POSTGRES_PASSWORD=1234", "POSTGRES_DB=optuna_study", "POSTGRES_USER=root"],
                ports={'5432/tcp': 5432},
                name="postgres",
                remove=True)
            container = self.client.containers.get("postgres")
            while "accepting connections" not in container.exec_run("pg_isready").output.decode():  # checks if db is up
                time.sleep(2)
            logger.info("DB-Container Running")

    def _shutdown_postgres_db(self):
        if current_process().name == "MainProcess":
            container = self.client.containers.get("postgres")
            container.stop()

    def single_process_run(self, study):
        objective = self.objective_cls(*self.objective_args)
        study.optimize(
            objective, n_trials=self.trials, timeout=None)

    def multi_process_run(self, trials, num_processes, study_name, storage_string, objective):
        # TODO: catch Cuda Errors in Logging
        # TODO: propper logging to file for multiprocessing?
        self._multi_process_run(trials, num_processes, study_name, storage_string, objective)

    def _multi_process_run(self, trials, num_processes, study_name, storage_string, objective):
        queue = Queue()
        for num_process_trial in self._split_trials(trials, num_processes):
            queue.put(num_process_trial)

        experiment_study_processes = []
        for idx in range(num_processes):
            process = ExperimentStudyProcess(queue, study_name, storage_string, objective)
            experiment_study_processes.append(process)
            process.start()
            time.sleep(.5)

        for experiment_job in experiment_study_processes:
            experiment_job.join()

    @staticmethod
    def _split_trials(n_trials, n_jobs):
        n_per_job, remaining = divmod(n_trials, n_jobs)
        for i in range(n_jobs):
            yield n_per_job + (1 if remaining > 0 else 0)
            remaining -= 1

    @staticmethod
    def multiprocess_optim_fun(study_name, storage_string, objective, trials):
        study = optuna.create_study(
            study_name=study_name, direction="maximize", storage=storage_string,
            load_if_exists=True)
        study.optimize(objective, n_trials=trials, timeout=None)

    def save_model_callback(self, study, trial):
        if study.best_trial.number == trial.number:
            pass

    def _run(self):
        # TODO: study failed error for value error at best_study
        # TODO: only one database startup for k>1
        if self.multiprocessing:
            try:
                self._start_postgres_db()
                study = optuna.create_study(
                    study_name=self.study_name, storage=self.storage_string, direction="maximize", load_if_exists=False)
                objective = self.objective_cls(*self.objective_args)
                self.multi_process_run(
                    self.trials, self.num_processes, self.study_name, self.storage_string, objective)
                best_trial = study.best_trial
                self._shutdown_postgres_db()
            except docker.errors.APIError:
                self._shutdown_postgres_db()
                logger.error("Docker Container Problem!", exc_info=True)
            except ValueError:
                logger.error("Best Trial could not be retrieved!", exc_info=True)
        else:
            study = optuna.create_study(
                study_name=self.study_name, direction="maximize")
            self.single_process_run(study)
            best_trial = study.best_trial
        return best_trial

    def run(self):
        best_trial = self._run()
        return best_trial


class ExperimentStudyProcess(Process):

    def __init__(self, queue, study_name, storage_string, objective):
        super().__init__()
        self.study_name = study_name
        self.storage_string = storage_string
        self.objective = objective
        self.queue = queue

    def run(self):
        while not self.queue.empty():
            trials = self.queue.get()
            study = optuna.create_study(
                study_name=self.study_name, direction="maximize", storage=self.storage_string,
                load_if_exists=True)
            study.optimize(self.objective, n_trials=trials, timeout=None)


if __name__ == "__main__":
    client = docker.from_env()
    client.containers.run(
        "postgres:14.1", detach=True,
        environment=["POSTGRES_PASSWORD=1234", "POSTGRES_DB=optuna_study", "POSTGRES_USER=root"],
        ports={'5432/tcp': 5432},
        name="postgres",
        remove=True)
    # time.sleep(5)
    # container = client.containers.get("postgres")
    # container.stop()
