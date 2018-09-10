import pandas as pd
import numpy as np

def create_asset_index(df):
    df['asset_index'] = (df['refrig'] +
                         df['computer'] +
                         (df['v18q1'] > 0) +
                         df['television'])
    return df


def create_housing_quality_features(df):
    df['wall_quality'] = 0 * df['epared1'] + 1 * df['epared2'] + 2 * df['epared3']
    df['roof_quality'] = 0 * df['etecho1'] + 1 * df['etecho2'] + 2 * df['etecho3']
    df['floor_quality'] = 0 * df['eviv1'] + 1 * df['eviv2'] + 2 * df['eviv3']
    return df


def preprocess(df):
    df = create_asset_index(df)
    df = create_housing_quality_features(df)

    return df


