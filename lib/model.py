import numpy as np
import pandas as pd

from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold

from .preprocessing import preprocess
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

full_pipeline = Pipeline(steps=[
                        ('features', feature_extraction),
                        ('imputer', Imputer(strategy='mean')),
                        ('feature_scaler', StandardScaler()),
                        ('random_forest', RandomForestClassifier(random_state=1))])

LEVELS = ['low', 'medium']
DEPTHS = [20, 30, 40]
param_grid = dict(features__extract_educ__level=LEVELS,
                 features__extract_rent__level=LEVELS,
                 features__extract_demog__level=LEVELS,
                 features__extract_houseq__level=LEVELS,
                 features__extract_housec__level=LEVELS,
                 features__extract_assets__level=LEVELS,
                 random_forest__max_depth=DEPTHS,
                 # svc__gamma=[0.01, 0.1,1,10],
                 # svc__C=[0.001,0.01,0.1,1,10,100,1000]
                 )


def load_and_process_training_data():
    """
    Loads training data
    :return: two pd.DataFrames, [X_train, y_train]
    """
    train = pd.read_csv('../input/train.csv')

    # IMPORTANT: according to the competition (https://www.kaggle.com/c/costa-rican-household-poverty-prediction/discussion/61403),
    # ONLY heads of household are used in scoring, so we should only train on them (with some household aggregate features)
    X_train = preprocess(train).drop('Target', axis=1).query('parentesco1 == 1')
    y_train = train.query('parentesco1 == 1')['Target']

    return X_train, y_train


if __name__ == 'main':
    # Run all steps
    pass
