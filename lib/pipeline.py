from sklearn.base import BaseEstimator, TransformerMixin


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


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom sklearn transformer for using a feature engineering function, with control over
    """
    def __init__(self, func, level='low'):
        """
        :param func: a function that returns a pd.DataFrame with features of interest
        :param level: str, one of ['low', 'medium', 'high'] passed to function
        """
        self.func = func
        self.level = level

    def fit(self, X, y=None):
        return self

    def transform(self, data):
        return self.func(data, self.level)
