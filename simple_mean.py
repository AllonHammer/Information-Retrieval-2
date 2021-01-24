import numpy as np
from interface import Regressor
from utils import get_data


class SimpleMean(Regressor):
    def __init__(self):
        self.user_means = None

    def fit(self, X: np.array):
        """ calc mean cross rows"""
        rows = X[:, 0]
        ratings = X[:, 2].astype(float)
        cnt = np.bincount(rows)
        sum = np.bincount(rows, weights = ratings)
        self.user_means = sum/cnt

    def predict_on_pair(self, user: int, item: int):
        if user == -999: #unknown user
            return 3.0 # (1+5)/2
        return float(self.user_means[user])


if __name__ == '__main__':
    train, validation = get_data()

    baseline_model = SimpleMean()
    baseline_model.fit(train)
    print(baseline_model.calculate_rmse(validation))
