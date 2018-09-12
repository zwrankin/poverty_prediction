import pandas as pd
import numpy as np


def create_asset_index(df):
    df.loc[df['v18q'] == 0, 'v18q1'] = 0  # num tablets = 0 if has_tablet == 0
    df['asset_index'] = (df['refrig'] +
                         df['computer'] +
                         (df['v18q1'] > 0) +
                         df['television'])
    return df


def create_housing_quality_features(df):
    df['wall_quality'] = 0 * df['epared1'] + 1 * df['epared2'] + 2 * df['epared3']
    df['roof_quality'] = 0 * df['etecho1'] + 1 * df['etecho2'] + 2 * df['etecho3']
    df['floor_quality'] = 0 * df['eviv1'] + 1 * df['eviv2'] + 2 * df['eviv3']
    df['house_material_vulnerability'] = df['epared1'] + df['etecho1'] + df['eviv1']
    return df


def create_housing_index(df):
    df['house_utility_vulnerability'] = (df['sanitario1'] +
                                       df['pisonotiene'] +
                                       (df['cielorazo'] == 0) +  # coding is opposite (the house has ceiling)
                                       df['abastaguano'] +
                                       df['noelec'] +
                                       df['sanitario1'])
    return df


def clean_dependency(df):
    df['calc_dependency'] = (df['hogar_nin'] + df['hogar_mayor']) / df['hogar_adul']
    df.loc[df['hogar_adul'] == 0, 'calc_dependency'] = 8  # this is hacky, but original dependency var is 8 when no adults
    df['calc_dependency_bin'] = np.where(df['calc_dependency'] == 0, 0, 1)
    return df


def clean_monthly_rent(df):
    # Fill in households that fully own house with 0 rent payment
    df.loc[(df['tipovivi1'] == 1), 'v2a1'] = 0  # tipovivi1, =1 own and fully paid house
    df.loc[df['v2a1'].isnull(), 'v2a1-missing'] = 1
    df['v2a1_missing'] = np.where(df['v2a1'].isnull(), 1, 0)
    return df


def clean_education(df):
    df['no_primary_education'] = df['instlevel1'] + df['instlevel2']
    df['rez_esc_scaled'] = df['rez_esc'] / (df['age'] - 6)  # years behind per year (ages 7-17)
    # Find the max (scaled) years behind in schooling per household
    df['hh_max_rez_esc_scaled'] = df.groupby('idhogar')['rez_esc_scaled'].transform(lambda x: x.max())
    return df


def per_capita_transformations(df):
    df['phones_pc'] = df['qmobilephone'] / df['tamviv']
    df['tablets_pc'] = df['v18q1'] / df['tamviv']
    df['rooms_pc'] = df['rooms'] / df['tamviv']
    df['rent_pc'] = df['v2a1'] / df['tamviv']
    return df


def map_booleans(df):
    mapping = {"yes": 1, "no": 0}
    df['edjefa'] = df['edjefa'].replace(mapping).astype(np.float64)
    df['edjefe'] = df['edjefe'].replace(mapping).astype(np.float64)
    return df


def preprocess(df):
    df = create_asset_index(df)
    df = create_housing_quality_features(df)
    df = create_housing_index(df)
    df = clean_dependency(df)
    df = clean_monthly_rent(df)
    df = clean_education(df)
    df = per_capita_transformations(df)
    # df = map_booleans(df)

    return df
