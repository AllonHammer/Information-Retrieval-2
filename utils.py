import pandas as pd
from config import TRAIN_PATH, VALIDATION_PATH, USER_COL_NAME_IN_DATAEST, ITEM_COL_NAME_IN_DATASET, RATING_COL_NAME_IN_DATASET



def get_data():
    """
    reads train, validation to python indices so we don't need to deal with it in each algorithm.
    of course, we 'learn' the indices (a mapping from the old indices to the new ones) only on the train set.
    if in the validation set there is an index that does not appear in the train set then we can put np.nan or
     other indicator that tells us that.
    """

    dypes = {USER_COL_NAME_IN_DATAEST: 'category', ITEM_COL_NAME_IN_DATASET: 'category', RATING_COL_NAME_IN_DATASET: int}
    train = pd.read_csv(TRAIN_PATH, dtype=dypes)
    valid = pd.read_csv(VALIDATION_PATH,  dtype=dypes)
    idx_to_user = dict(enumerate(train[USER_COL_NAME_IN_DATAEST].cat.categories))
    user_to_idx = {v: k for k, v in idx_to_user.items()}
    idx_to_item = dict(enumerate(train[ITEM_COL_NAME_IN_DATASET].cat.categories))
    item_to_idx = {v: k for k, v in idx_to_item.items()}
    for ds in (train, valid):
        ds[USER_COL_NAME_IN_DATAEST] = ds[USER_COL_NAME_IN_DATAEST].apply(lambda x: user_to_idx[x] if x in user_to_idx else -999)
        ds[ITEM_COL_NAME_IN_DATASET] = ds[ITEM_COL_NAME_IN_DATASET].apply(lambda x: item_to_idx[x] if x in item_to_idx else -999)


    return train.values.astype(int), valid.values.astype(int)






class Config:
    def __init__(self, **kwargs):
        self._set_attributes(kwargs)

    def _set_attributes(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)








