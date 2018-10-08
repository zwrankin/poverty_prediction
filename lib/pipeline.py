import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.utils import resample
from lightgbm import LGBMClassifier


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
    Custom sklearn transformer for using a feature engineering function, with args accessible via model tuning
    """
    def __init__(self, func, level='low', drop_correlated_features=False):
        """
        :param func: a function that returns a pd.DataFrame with features of interest
        :param level: str, one of ['low', 'medium', 'high'] passed to function
        """
        self.func = func
        self.level = level
        self.drop_correlated_features = drop_correlated_features

    def fit(self, X, y=None):
        return self

    def transform(self, data):
        return self.func(data, self.level, self.drop_correlated_features)


def bag_and_boost_model(X, y, random_state, early_stopping_rounds=10):
    """
    Fits a gradient boosted model to bootstrapped data, and uses the unsampled (aka "out-of-bag") observations
    as a validation set for early stopping
    Different than sklearn's native BaggingClassifier because it allows early stopping
    """

    # First implementation is cleaner with pd.DataFrame
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(y, np.ndarray):
        X = pd.DataFrame(y)

    # Note that due to bootstrapping, sample_fraction=1 still leaves ~37% of data in the validation set
    X_train, y_train = resample(X, y, n_samples=len(X), replace=True, random_state=random_state)
    valid_idx = [i for i in X.index if i not in X_train.index]
    X_valid = X.loc[valid_idx]
    y_valid = y.loc[valid_idx]

    tuned_params = {'boosting_type': 'dart',
                    'colsample_bytree': 0.5796397953791418,
                    'learning_rate': 0.08739537002929919,
                    'min_child_samples': 15,
                    'num_leaves': 48,
                    'reg_alpha': 0.4239159481112283,
                    'reg_lambda': 0.36419362906439723,
                    'subsample_for_bin': 40000,
                    'subsample': 0.986210861412967,
                    'class_weight': 'balanced',
                    'limit_max_depth': 1,
                    'max_depth': 22}

    fit_params = {"eval_set": [(X_valid, y_valid)],
                  "early_stopping_rounds": early_stopping_rounds,
                  "verbose": False}

    clf = LGBMClassifier(**tuned_params, objective='multiclass',
                         n_jobs=-1, n_estimators=10000,
                         random_state=random_state)

    return clf.fit(X_train, y_train, **fit_params)


class BaggedLGBMClassifier(BaseEstimator, ClassifierMixin):
    """ADD DOCSTRING"""

    def __init__(self, n_meta_estimators=5, early_stopping_rounds=10):
        self.n_meta_estimators = n_meta_estimators
        self.early_stopping_rounds = early_stopping_rounds

    def fit(self, X, y=None):
        if self.n_meta_estimators != 5:
            raise NotImplementedError('sorry, only 5 meta-estimators supported now')

        clf1 = bag_and_boost_model(X, y, early_stopping_rounds=self.early_stopping_rounds, random_state=1)
        clf2 = bag_and_boost_model(X, y, early_stopping_rounds=self.early_stopping_rounds, random_state=2)
        clf3 = bag_and_boost_model(X, y, early_stopping_rounds=self.early_stopping_rounds, random_state=3)
        clf4 = bag_and_boost_model(X, y, early_stopping_rounds=self.early_stopping_rounds, random_state=4)
        clf5 = bag_and_boost_model(X, y, early_stopping_rounds=self.early_stopping_rounds, random_state=5)

        self.clfs = [clf1, clf2, clf3, clf4, clf5]

        return self

    def predict(self, X, y=None):
        probs = np.mean([clf.predict_proba(X) for clf in self.clfs], axis=0)
        predictions = pd.DataFrame(probs).idxmax(axis=1)
        return np.array(predictions)
