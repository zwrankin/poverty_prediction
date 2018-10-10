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


class BaggedLGBMClassifier(BaseEstimator, ClassifierMixin):
    """
    Custom sklearn classifier that uses bagging to fit gradient boosted model with early stopping
    Early stopping uses the unsampled (aka "out-of-bag") observations as validation set
    Different than sklearn's native BaggingClassifier because it allows early stopping
    """

    # TODO - these defaults are from old optimization, not sure if it would be better to use LGBMClassifier defaults
    def __init__(self, n_meta_estimators=5, early_stopping_rounds=10, random_state=0,
                 boosting_type='dart', colsample_bytree=0.58, learning_rate=0.087, min_child_samples=15,
                 num_leaves=48, reg_alpha=0.42, reg_lambda=0.36, subsample_for_bin=40000, subsample=0.99,
                 class_weight='balanced', limit_max_depth=1, max_depth=22):

        self.n_meta_estimators = n_meta_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.boosting_type = boosting_type
        self.colsample_bytree = colsample_bytree
        self.learning_rate = learning_rate
        self.min_child_samples = min_child_samples
        self.num_leaves = num_leaves
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.subsample_for_bin = subsample_for_bin
        self.subsample = subsample
        self.class_weight = class_weight
        self.limit_max_depth = limit_max_depth
        self.max_depth = max_depth

    def bag_and_boost_model(self, X, y, random_state):
        """
        Fits single gradient boosted model to bootstrapped data, using unsampled (aka "out-of-bag") observations
        as a validation set for early stopping
        """

        # First implementation is cleaner with pd.DataFrame (as opposed to np.array)
        # Howevever, during some cross-validation processes, I was finding some inconsistency between
        # preservation of indices between X and y (e.g. because X became np.array in pipeline but y stayed pd.DataFrame)
        X = pd.DataFrame(X).reset_index().drop('index', axis=1)
        y = pd.DataFrame(y).reset_index().drop('index', axis=1)
        assert (y.index == X.index, 'X and y indices do not match')

        # Note that due to bootstrapping, sample_fraction=1 still leaves ~37% of data in the validation set
        X_train, y_train = resample(X, y, n_samples=len(X), replace=True, random_state=random_state)
        valid_idx = [i for i in X.index if i not in X_train.index]
        X_valid = X.loc[valid_idx]
        y_valid = y.loc[valid_idx]

        lgb_params = {'boosting_type': self.boosting_type,
                      'colsample_bytree': self.colsample_bytree,
                      'learning_rate': self.learning_rate,
                      'min_child_samples': self.min_child_samples,
                      'num_leaves': self.num_leaves,
                      'reg_alpha': self.reg_alpha,
                      'reg_lambda': self.reg_lambda,
                      'subsample_for_bin': self.subsample_for_bin,
                      'subsample': self.subsample,
                      'class_weight': self.class_weight,
                      'limit_max_depth': self.limit_max_depth,
                      'max_depth': self.max_depth}

        fit_params = {"eval_set": [(X_valid, y_valid)],
                      "early_stopping_rounds": self.early_stopping_rounds,
                      "verbose": False}

        clf = LGBMClassifier(**lgb_params, objective='multiclass',
                             n_jobs=-1, n_estimators=10000,
                             random_state=random_state)

        return clf.fit(X_train, y_train, **fit_params)

    def fit(self, X, y=None):
        if self.n_meta_estimators != 5:
            raise NotImplementedError('sorry, only 5 meta-estimators supported now')

        clf1 = self.bag_and_boost_model(X, y, random_state=self.random_state + 1)
        clf2 = self.bag_and_boost_model(X, y, random_state=self.random_state + 2)
        clf3 = self.bag_and_boost_model(X, y, random_state=self.random_state + 3)
        clf4 = self.bag_and_boost_model(X, y, random_state=self.random_state + 4)
        clf5 = self.bag_and_boost_model(X, y, random_state=self.random_state + 5)
        self.clfs = [clf1, clf2, clf3, clf4, clf5]

        self.feature_importances_ = np.mean([c.feature_importances_ for c in self.clfs], axis=0)
        self.feature_importances_ /= self.feature_importances_.max()  # scale 0-1

        return self

    def predict(self, X, y=None):
        probs = np.mean([clf.predict_proba(X) for clf in self.clfs], axis=0)
        predictions = pd.DataFrame(probs).idxmax(axis=1)
        return np.array(predictions)
