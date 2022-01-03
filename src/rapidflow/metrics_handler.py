from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve, auc
import numpy as np
from collections import defaultdict
import pandas as pd


class MetricsHandler:

    def __init__(self) -> None:
        """
        Serves as a basic class for generating metrics for predicted results and scores.
        """
        pass

    def calculate_classification_metrics(self, targets, predictions):
        classifcation_results = classification_report(
            targets, predictions, output_dict=True)
        classifcation_results = pd.json_normalize(classifcation_results, sep=' ').to_dict(orient="records")[0]
        return classifcation_results

    def calculate_anomaly_metrics(self, targets, anomaly_score):
        aupr_score = self.aupr(targets, anomaly_score)
        base = self.aupr_base_rate(targets)
        relative_aupr_base_diff = (aupr_score-base)/(1-base)

        return {
            "AUC": roc_auc_score(targets, anomaly_score),
            "AUPR": aupr_score,
            "AUPR-Base": base,
            "R-AUPR-Base-Diff": relative_aupr_base_diff
        }

    def calculate_all_metrics(self, targets, predictions, anomaly_score):
        classification_metrics = self.calculate_classification_metrics(targets, predictions)
        anomaly_metrics = self.calculate_anomaly_metrics(targets, anomaly_score)
        classification_metrics.update(anomaly_metrics)
        return classification_metrics

    def average_results(self, test_results):
        averaged_results = defaultdict(list)
        test_results = [test_results for test_results in test_results if not None]
        for test_result in test_results:
            for key, value in test_result.items():
                averaged_results[key].append(value)
        for key, value in averaged_results.items():
            averaged_results[key] = self._mean_std(value)
        return averaged_results

    def _mean_std(self, value_list):
        mean = np.mean(value_list)
        std = np.std(value_list)
        return f"{mean:.2f} ({std:.2f})"

    def aupr(self, targets, logits):
        """
        AUPR = Area under Precision Recall Curve.

        Args:
            targets ([type]): [description]
            logits ([type]): [description]

        Returns:
            [type]: [description]
        """
        precision, recall, _ = precision_recall_curve(targets, logits)
        return auc(recall, precision)

    def aupr_base_rate(self, targets):
        """
        Calculates the static AUPR Base Rate.

        Args:
            targets ([type]): [description]

        Returns:
            [type]: [description]
        """
        return len(targets[targets == 1]) / len(targets)
