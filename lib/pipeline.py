
from sklearn.metrics import f1_score, make_scorer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from.preprocessing import preprocess

# Custom scorer for cross validation
f1_scorer = make_scorer(f1_score, greater_is_better=True, average='macro')

kfold = StratifiedKFold(n_splits=5, random_state=1)

class LazyProcessing(BaseEstimator, TransformerMixin):
    """
    Making the preprocessing script a step in the pipeline. This will be gradually split into
    individual transformers.
    """
    def __init__(self):
        self.preprocess = preprocess

    def fit(self, X, y=None):
        return self

    def transform(self, data):
        return preprocess(data)


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


# Here are some hard-coded things use across notebooks
TEST_FEATURES = ['meaneduc', 'no_primary_education', 'hh_max_rez_esc_scaled'] + [
                'wall_quality', 'roof_quality', 'floor_quality', 'house_material_vulnerability'] + [
                'house_utility_vulnerability', 'sanitario1', 'pisonotiene', 'cielorazo', 'abastaguano',
                'noelec', 'sanitario1'] + [
                'asset_index', 'v18q', 'v18q1', 'refrig', 'computer', 'television'] + [
                'tamviv', 'hogar_nin', 'overcrowding'] + [
                'calc_dependency', 'calc_dependency_bin'] + [
                'phones_pc', 'tablets_pc', 'rooms_pc', 'rent_pc'] + [
                'v2a1', 'v2a1_missing']

TRANSFORMER_PIPELINE = Pipeline(steps=[
                        ('feature_extraction', FeatureExtractor(TEST_FEATURES)),
                        ('imputer', Imputer(strategy='mean')),
                        ('feature_scaler', StandardScaler()),
                        # ('feature_selection', SelectFromModel(ExtraTreesClassifier())),
                        ])
