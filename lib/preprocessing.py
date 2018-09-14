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
    # Original 'dependency' variable has 'yes' and 'no', so better to calculate it per definition
    df['calc_dependency'] = (df['hogar_nin'] + df['hogar_mayor']) / df['hogar_adul']
    df.loc[df['hogar_adul'] == 0, 'calc_dependency'] = 8  # this is hacky, but original dependency var is 8 when no adults
    df['calc_dependency_bin'] = np.where(df['calc_dependency'] == 0, 0, 1)
    return df


def clean_monthly_rent(df):

    # Fill in households that fully own house with 0 rent payment
    df.loc[(df['tipovivi1'] == 1), 'v2a1'] = 0  # tipovivi1, =1 own and fully paid house
    df['v2a1_missing'] = np.where(df['v2a1'].isnull(), 1, 0)

    df['rent_by_hhsize'] = df['v2a1'] / df['hhsize']  # rent by household size
    df['rent_by_people'] = df['v2a1'] / df['r4t3']  # rent by people in household
    df['rent_by_rooms'] = df['v2a1'] / df['rooms']  # rent by number of rooms
    df['rent_by_living'] = df['v2a1'] / df['tamviv']  # rent by number of persons living in the household
    df['rent_by_minor'] = df['v2a1'] / df['hogar_nin']
    df['rent_by_adult'] = df['v2a1'] / df['hogar_adul']
    df['rent_by_dep'] = df['v2a1'] / df['calc_dependency']
    # df['rent_by_educ'] = df['v2a1'] / df['meaneduc']
    # df['rent_by_numPhone'] = df['v2a1'] / df['qmobilephone']
    # df['rent_by_gadgets'] = df['v2a1'] / (df['computer'] + df['mobilephone'] + df['v18q'])
    # df['rent_by_num_gadgets'] = df['v2a1'] / (df['v18q1'] + df['qmobilephone'])
    # df['rent_by_appliances'] = df['v2a1'] / df['appliances']

    #Top code at #1 million per whatever (mostly to avoid inf
    for col in ['rent_by_minor', 'rent_by_adult', 'rent_by_dep']:
        df.loc[df[col] > 1000000, col] = 1000000
    return df


def clean_education(df):
    # A few individuals (incl heads of hh) have missing meaneduc but nonmissing escolari
    df.loc[df['meaneduc'].isnull(), 'meaneduc'] = df['escolari']

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


def fit_outliers(df):
    # According to competition host (https://www.kaggle.com/c/costa-rican-household-poverty-prediction/discussion/61403)
    # we can safely change the rez_esc value to 5
    df.loc[df['rez_esc'] == 99.0, 'rez_esc'] = 5
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
    df = fit_outliers(df)

    # Drop object columns that are not necessary for model
    df.drop(['Id', 'idhogar', 'dependency', 'edjefe', 'edjefa'], axis=1, inplace=True)

    return df


def load_and_process_training_data():
    """
    (somewhat hacky) Convenience function to clean up notebooks
    :return: two pd.DataFrames, [X_train, y_train]
    """
    train = pd.read_csv('../input/train.csv')

    X_train = preprocess(train).drop('Target', axis=1).query(
        'parentesco1 == 1')  # try subsetting to ONLY train on head of household
    y_train = train.query('parentesco1 == 1')['Target']  # try subsetting to ONLY train on head of household

    return X_train, y_train

