
from sklearn.metrics import f1_score, make_scorer

# Custom scorer for cross validation
f1_scorer = make_scorer(f1_score, greater_is_better=True, average='macro')

class FeatureExtractor(object):

    def __init__(self, cols):
        self.cols = cols

    def transform(self, data):
        return data[self.cols]

    def fit(self, X, y=None):
        return self
