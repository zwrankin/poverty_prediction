import numpy as np
import pandas as pd

from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold

from .preprocessing import preprocess
from .preprocessing import run_feature_engineering
from .preprocessing import feature_engineer_rent, feature_engineer_education, feature_engineer_demographics
from .preprocessing import feature_engineer_house_rankings, feature_engineer_house_characteristics, feature_engineer_assets
from .pipeline import FeatureEngineer

# Custom scorer for cross validation
f1_scorer = make_scorer(f1_score, greater_is_better=True, average='macro')

kfold = StratifiedKFold(n_splits=5, random_state=1)

feature_extraction = FeatureUnion(transformer_list=[
    ('extract_rent', FeatureEngineer(feature_engineer_rent)),
    ('extract_educ', FeatureEngineer(feature_engineer_education)),
    ('extract_demog', FeatureEngineer(feature_engineer_demographics)),
    ('extract_houseq', FeatureEngineer(feature_engineer_house_rankings)),
    ('extract_housec', FeatureEngineer(feature_engineer_house_characteristics)),
    ('extract_assets', FeatureEngineer(feature_engineer_assets))
])

transformer_pipeline = Pipeline(steps=[
                        ('features', feature_extraction),
                        ('imputer', Imputer(strategy='mean')),
                        ('feature_scaler', StandardScaler()),
                        ])

pipeline_rf = Pipeline(steps=[
                        ('features', feature_extraction),
                        ('imputer', Imputer(strategy='mean')),
                        ('feature_scaler', StandardScaler()),
                        ('random_forest', RandomForestClassifier(random_state=1))])

LEVELS = ['low', 'medium']
DEPTHS = [20, 30, 40]
param_grid_rf = dict(features__extract_educ__level=LEVELS,
                 features__extract_rent__level=LEVELS,
                 features__extract_demog__level=LEVELS,
                 features__extract_houseq__level=LEVELS,
                 features__extract_housec__level=LEVELS,
                 features__extract_assets__level=LEVELS,
                 random_forest__max_depth=DEPTHS,
                 )


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


if __name__ == 'main':
    # Run all steps
    pass
