import pandas as pd

from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score

from .preprocessing import preprocess
from .preprocessing import run_feature_engineering
from .pipeline import BaggedLGBMClassifier

# Custom scorer for cross validation
f1_scorer = make_scorer(f1_score, greater_is_better=True, average='macro')

kfold = StratifiedKFold(n_splits=5, random_state=10)

clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=1)
pipeline = Pipeline(steps=[
                        ('imputer', Imputer(strategy='mean')),
                        ('feature_scaler', StandardScaler()),
                        ('selector', SelectFromModel(clf, threshold=0.012)),
                        ('model', BaggedLGBMClassifier(random_state=10)),
                        ])


def load_and_process_training_data(engineer_features=True, subset_to_hh=True):
    """
    Convenience function
    :param engineer_features: whether to run feature engineering
    :param subset_to_hh: whether to subset to heads of household
    :return:  pd.DataFrames, [X_train, y_train]
    """

    train = pd.read_csv('../input/train.csv')
    hh_idx = train.parentesco1 == 1

    X_train = preprocess(train).drop('Target', axis=1)
    if engineer_features:
        X_train = run_feature_engineering(X_train, level='medium')

    y_train = train['Target']
    if subset_to_hh:
        X_train = X_train.loc[hh_idx]
        y_train = y_train.loc[hh_idx]

    return X_train, y_train


def load_and_process_test_data(engineer_features=True):
    """
    Convenience function
    :param engineer_features: whether to run feature engineering
    :return: pd.DataFrame X_test
    """
    test = pd.read_csv('../input/test.csv')
    X_test = preprocess(test)
    if engineer_features:
        X_test = run_feature_engineering(X_test, level='medium')
    return X_test


def pipeline_cv_score(model_pipeline, X, y):
    """Display cross-validation score of a pipeline given data X and y"""
    cv_score = cross_val_score(model_pipeline, X, y, cv=kfold, scoring=f1_scorer, n_jobs=-1)
    print(f'Cross Validation F1 Score = {round(cv_score.mean(), 4)} with std = {round(cv_score.std(), 4)}')


if __name__ == 'main':
    # Run all steps
    pass
