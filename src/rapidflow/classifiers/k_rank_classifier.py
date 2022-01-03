import numpy as np


class KRankClassifier:

    def __init__(self, k) -> None:
        # TODO: optimization strategy for K
        """
        Takes the first k inputss and classifies them as with label '1'.

        Args:
            k ([type]): [description]
        """
        self.k = k

    def rank_classify(self, scores):
        """
        Sorts all values and takes the first k values for the '1' class.
        All values below are assigned with '0'.

        Args:
            scores ([type]): [description]

        Returns:
            [type]: [description]
        """
        indices = scores[::-1].argsort()
        total_k = int(indices.shape[0]*self.k)
        predictions = np.concatenate([
            np.ones(total_k), np.zeros(scores.shape[0]-total_k)])
        predictions = predictions[indices]
        return predictions


if __name__ == "__main__":

    scores = np.array(list(range(20)))
    classifier = KRankClassifier(0.5)
    predictions = classifier.rank_classify(scores)
    print(predictions)
    print(scores)
