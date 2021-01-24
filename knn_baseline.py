import numpy as np
import pickle
from utils import get_data, Config
from knn import KnnItemSimilarity
from config import BASELINE_PARAMS_FILE_PATH

class KnnBaseline(KnnItemSimilarity):
    def __init__(self, config):
        super(KnnBaseline, self).__init__(config)
        with open(BASELINE_PARAMS_FILE_PATH, 'rb') as handle:
            baselines = pickle.load(handle)
        self.global_bias = baselines['global_bias']
        self.user_biases = baselines['user_biases']
        self.item_biases = baselines['item_biases']



    def predict_on_pair(self, user: int, item: int):
        """
        given user u and item i, this function calculates
        N_uk(i)= the correlations of k nearest neighbors of item i that are rated by user u
        R_uj = the rating of these items
        B_ujs = the bias of these items with user u
        b_ui = pretrained bias of user u and item i

        the output is then N_uk(i).dot(R_uj-Buj) / ∑ N_uk(i)   + b_ui
        """

        # if a user or an item is unknown we use the same logic like in linear_regression_baseline
        if user == -999 and item != -999:
            return self.global_bias +  self.item_biases[item]
        if user != -999 and item == -999:
            return self.global_bias + self.user_biases[user]
        if user == -999 and item == -999:
            return self.global_bias

        csr =self.csr
        b_ui = self.global_bias + self.user_biases[user] +self.item_biases[item] #bias for user u and item i
        items_ranked_by_user = csr.indices[csr.indptr[user]: csr.indptr[user+1]].tolist() #all items that user u ranked
        corrs = self.item_corr[item] # all correalations with item i. These are tuples (item_id, corr)
        # we omit the first item since it has a correlation with itself (corr=1)
        # we omit all items that were not ranked by the user
        # we omit all negative correlations
        # we take the top k that remain
        top_k_corrs = [pair for pair in corrs[1:] if pair[0] in items_ranked_by_user and pair[1]>0][:self.k]

        #biases for the ranked items of user u
        B_ujs = [self.global_bias+self.user_biases[user]+self.item_biases[pair[0]] for pair in top_k_corrs]
        N_uks = [pair[1] for pair in top_k_corrs] #the corr of k nearest neighbors of item i that are rated by user u
        R_ujs = [csr[user, pair[0]] for pair in top_k_corrs] #ratings of these items

        B_ujs = np.array(B_ujs)
        N_uks = np.array(N_uks)
        R_ujs = np.array(R_ujs)

        pred = (np.dot(N_uks, (R_ujs.T-B_ujs)) / N_uks.sum()) +b_ui
        return pred




if __name__ == '__main__':
    baseline_knn_config = Config(k=10)
    train, validation = get_data()
    knn_baseline = KnnBaseline(baseline_knn_config)
    knn_baseline.fit(train)
    print(knn_baseline.calculate_rmse(validation))
