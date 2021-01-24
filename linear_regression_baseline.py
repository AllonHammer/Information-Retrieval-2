import numpy as np
import pickle
from interface import Regressor
from utils import Config, get_data
from config import BASELINE_PARAMS_FILE_PATH
from typing import Dict

class Baseline(Regressor):
    def __init__(self, config):
        self.lr = config.lr
        self.gamma = config.gamma
        self.train_epochs = config.epochs
        self.n_users = None
        self.n_items = None
        self.user_biases = None  # b_u (users) vector
        self.item_biases = None  # # b_i (items) vector
        self.current_epoch = 0
        self.global_bias = None

    def record(self, covn_dict: Dict):
        print('Epoch {} Out of {}'.format(self.current_epoch+1, self.train_epochs))
        print(covn_dict)
        print('*********************')


    def calc_regularization(self):
        """
        calc  γ · ( Σ u b u 2 + Σ i b i 2 )
        """
        return self.gamma * ((self.user_biases ** 2).sum() + (self.item_biases ** 2).sum())

    def fit(self, X: np.array):
        """
        Initializes biases and starts training epochs.
        Prints logs along the way and saves the trained weights

        """

        self.n_users = len(np.unique(X[:, 0]))
        self.n_items = len(np.unique(X[:, 1]))
        self.user_biases = np.zeros(self.n_users)
        self.item_biases = np.zeros(self.n_items)
        self.global_bias = np.mean(X[: ,2])
        while self.current_epoch < self.train_epochs:
            self.run_epoch(X)
            train_mse = np.square(self.calculate_rmse(X))
            train_objective = train_mse * X.shape[0] + self.calc_regularization()
            epoch_convergence = {"train_objective": np.round(train_objective, 2),
                                 "train_mse":       np.round(train_mse,2)}
            self.record(epoch_convergence)
            self.current_epoch += 1
        self.save_params()




    def run_epoch(self, X: np.array):
        """
        update trainable params with SGD (batch_size=1)

        """
        np.random.shuffle(X)
        for row in X:
            user, item, rating = row
            err = rating - self.predict_on_pair(user, item)
            # update step with SGD, the sign is + because of the derivative chain rule
            self.user_biases[user] += self.lr * (err - self.gamma * self.user_biases[user])
            self.item_biases[item] += self.lr * (err - self.gamma * self.item_biases[item])


    def predict_on_pair(self, user: int, item: int):
        if user == -999 and item != -999:
            return self.global_bias +  self.item_biases[item]
        if user != -999 and item == -999:
            return self.global_bias + self.user_biases[user]
        if user == -999 and item == -999:
            return self.global_bias

        return self.global_bias + self.user_biases[user] + self.item_biases[item]

    def save_params(self):
        baseline_params = {'user_biases': self.user_biases,
                           'item_biases': self.item_biases,
                           'global_bias': self.global_bias}
        with open(BASELINE_PARAMS_FILE_PATH, 'wb') as handle:
            pickle.dump(baseline_params, handle, protocol=pickle.HIGHEST_PROTOCOL)




if __name__ == '__main__':
    baseline_config = Config(
        lr=0.001,
        gamma=0.001,
        epochs=10)

    train, validation = get_data()
    baseline_model = Baseline(baseline_config)
    baseline_model.fit(train)
    print(baseline_model.calculate_rmse(validation))
