
from sklearn.metrics import f1_score, make_scorer
from sklearn.base import BaseEstimator, TransformerMixin

# Custom scorer for cross validation
f1_scorer = make_scorer(f1_score, greater_is_better=True, average='macro')

class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom sklearn transformer to select certain features from a pd.DataFrame
    NOTE - as it needs a pd.DataFrame, this should be the first tranformer or be used after one
    that produces a pd.DataFrame (ie, not an array)
    param: cols - list of features
    """
    def __init__(self, cols):
        if isinstance(cols, str): cols = [cols]  ## if single feature given as string
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, data):
        return data[self.cols]
