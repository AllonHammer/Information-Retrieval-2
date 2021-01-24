import numpy as np
import pandas as pd
import time
from interface import Regressor
from utils import get_data, Config
from scipy.sparse import csr_matrix
from config import CORRELATION_PARAMS_FILE_PATH, CSV_COLUMN_NAMES
from pathlib import Path
from tqdm import tqdm

class KnnItemSimilarity(Regressor):
    def __init__(self, config):
        self.k = config.k
        self.item_corr = {}
        self.X = None

    def fit(self, X: np.array):
        """
        calculates (or loads) item to item corr matrix
        updates top_k corr dict
        """
        self.csr = self.to_csr(X)  #save original data as csr

        if Path(CORRELATION_PARAMS_FILE_PATH).exists():
            print('loading pretrained correltions')
            corr_matrix = self.upload_params()
            print('pretrained correlations loaded')
        else:
            print('estimating correlations')
            start_time = time.time()
            corr_matrix = self.build_item_to_itm_corr()
            elapsed_time = time.time() - start_time
            print('It took ~ {} seconds to estimate the correlations'.format(round(elapsed_time)))
            self.save_params(corr_matrix)

        self.top_k_corr_dict(corr_matrix)




    def build_item_to_itm_corr(self):
        """
        calculates the item-item correlation matrix in a clever way using csr matrix

        """
        csr = self.csr.T #we want correlations between items
        # calculate covariance matrix
        n = csr.shape[1] # number of users
        sum_cols = csr.sum(axis=1)  # shape I X 1
        means = sum_cols.dot(sum_cols.T.conjugate()) / n # I X I matrix [i1_mean * i1_mean, i1_mean * i2_mean ...
                                             #             i2_mean * i1_mean, i2_mean * i2_mean ... ]
        cov_matrix = (csr.dot(csr.T.conjugate()) - means) / (n-1) # IXI matix  [ cov(i1,i1), cov(i1,i2) ....
                                                                 #             cov(i2,i1), cov(i2,i2) ...]

        # calcualte corr matrix
        d = np.diag(cov_matrix)
        stddev = np.sqrt(np.outer(d, d))
        corr_matrix = cov_matrix/stddev
        corr_matrix = np.squeeze(np.asarray(corr_matrix)) #convert from np.matrix to np.array

        return corr_matrix


    def to_csr(self, X:np.array):
        """
        converts triplets of [user, item, rating] to sparse matrix representation

        """
        rows = X[:, 0]
        cols = X[:, 1]
        ratings = X[:, 2]
        # turn to sparse matrix representation
        csr = csr_matrix((ratings, (rows, cols)))
        return csr


    def top_k_corr_dict(self, corr_matrix: np.array):
        """
        given an item-item correlation matrix, computes a dict of top K correlated items for each item (not including self)
        {item_id : [(item_most_correlated, correlation), (item_2nd_most_correlated, correlation), ...(item_k_most_correlated, correlation)]}

        """
        print('Creating top K corr dict')
        for idx, row in tqdm(enumerate(corr_matrix)): # iterate row of corr matrix
            self.item_corr[idx] = []
            sorted_items = row.flatten().argsort()[::-1]
            sorted_corrs = row[sorted_items] # take the corr values
            for pair in zip(sorted_items, sorted_corrs): # append to dict
                self.item_corr[idx].append(pair)


    def predict_on_pair(self, user, item):
        """
        given user u and item i, this function calculates
        N_uk(i)= the correlations of k nearest neighbors of item i that are rated by user u
        R_uj = the rating of these items

        the output is then N_uk(i).dot(R_uj) / ∑ N_uk(i)
        """

        if user == -999 or item == -999: #since we dont have any biases trained, if this is an unknown user or item we will give a mean prediction of 3
            return 3.0

        csr =self.csr
        items_ranked_by_user = csr.indices[csr.indptr[user]: csr.indptr[user+1]].tolist() #all items that user u ranked
        corrs = self.item_corr[item] # all correalations with item i. These are tuples (item_id, corr)
        # we omit the first item since it has a correlation with itself (corr=1)
        # we omit all items that were not ranked by the user
        # we omit all negative correlations
        # we take the top k that remain
        top_k_corrs = [pair for pair in corrs[1:] if pair[0] in items_ranked_by_user and pair[1]>0][:self.k]
        N_uks = [pair[1] for pair in top_k_corrs] #the corr of k nearest neighbors of item i that are rated by user u
        R_ujs = [csr[user, pair[0]] for pair in top_k_corrs] #ratings of these items


        N_uks = np.array(N_uks)
        R_ujs = np.array(R_ujs)

        pred = np.dot(N_uks, R_ujs.T) / N_uks.sum()
        return pred

    def upload_params(self):
        """
        uploads correlations (item1 , item2, corr) from CSV --> corr matrix (IXI)
        """
        df = pd.read_csv(CORRELATION_PARAMS_FILE_PATH)
        #turn df with triplets (item1, item2, corr) to corr matrix
        corr_matrix = pd.pivot_table(df, values = CSV_COLUMN_NAMES[2], index= CSV_COLUMN_NAMES[0], columns=CSV_COLUMN_NAMES[1] ).values
        return corr_matrix

    def save_params(self, array: np.array):
        """
        saves corr matrix (IXI) --> CSV (item1, item2, corr)
        """
        df = pd.DataFrame(array).unstack().reset_index()  #turn corr matrix to df with triplets (item1, item2, corr)
        df.columns = CSV_COLUMN_NAMES
        convert_dict = {CSV_COLUMN_NAMES[0]: 'int16', CSV_COLUMN_NAMES[1]: 'int16', CSV_COLUMN_NAMES[2]: 'float16'}
        df = df.astype(convert_dict)
        df.to_csv(CORRELATION_PARAMS_FILE_PATH, index=False)


if __name__ == '__main__':
    knn_config = Config(k=25)
    train, validation = get_data()


    knn = KnnItemSimilarity(knn_config)


    knn.fit(train)
    print(knn.calculate_rmse(validation))

