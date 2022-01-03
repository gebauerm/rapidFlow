from scipy import optimize
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import numpy as np


class ThresholdBruteForceClassifier:

    scoring_function_registry = {
        "f1_score": f1_score,
        "accuracy": accuracy_score,
        "auc": roc_auc_score
    }

    def __init__(self, scoring_func_string="f1_score", threshold_steps_rate=.2) -> None:
        self.scoring_func = self.scoring_function_registry[scoring_func_string]
        self.threshold_steps_rate = threshold_steps_rate
        self.threshold = 0

    def brute_force(self, targets, loss):
        # TODO: get rid of torch tensors here
        threshold_steps = (np.max(loss).item() - np.min(loss).item()) * self.threshold_steps_rate
        if threshold_steps == 0:  # Not the best but works
            self.threshold = 0
            return None
        threshold_slice = slice(np.min(loss).item(), np.max(loss).item(), threshold_steps)
        threshold_range = (threshold_slice,)
        brute_force = optimize.brute(
            self._target_function, threshold_range, args=(loss, targets), full_output=True, finish=optimize.fmin
            )
        self.threshold = brute_force[0]
        return None

    @staticmethod
    def _threshold_classify(loss, threshold):
        """
        0 = Majority
        1 = Minority

        Args:
            loss ([type]): [description]
            threshold ([type]): [description]

        Returns:
            [type]: [description]
        """
        try:
            predicted_classes = (loss > threshold).astype(int)
        except TypeError:
            threshold = threshold[0]
            predicted_classes = (loss > threshold).astype(int)
        return predicted_classes

    def _target_function(self, threshold, *args):
        loss, targets = args
        predicted_classes = self._threshold_classify(loss, threshold)
        metric = -self.scoring_func(targets, predicted_classes)
        return metric

    def classify(self, anomaly_score):
        if self.threshold:
            predictions = self._threshold_classify(anomaly_score, self.threshold)
        else:
            raise AttributeError("No Threshold set!")
        return predictions
