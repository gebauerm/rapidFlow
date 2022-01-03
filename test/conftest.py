
import pytest as pt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


@pt.fixture
def generate_test_data():
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=.2)
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=.2)
    return (train_x, train_y), (val_x, val_y), (test_x, test_y)
