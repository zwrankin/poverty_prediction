import pandas as pd
import numpy as np


# Next three functions are from https://www.kaggle.com/gaxxxx/exploratory-data-analysis-lightgbm
# TODO - check validity
def fill_roof_exception(x):
    if (x['techozinc'] == 0) and (x['techoentrepiso'] == 0) and (x['techocane'] == 0) and (x['techootro'] == 0):
        return 1
    else:
        return 0

def fill_no_electricity(x):
    if (x['public'] == 0) and (x['planpri'] == 0) and (x['noelec'] == 0) and (x['coopele'] == 0):
        return 1
    else:
        return 0

def clean_dummy_features(df):
    df['roof_waste_material'] = df.apply(lambda x: fill_roof_exception(x), axis=1)
    df['electricity_other'] = df.apply(lambda x: fill_no_electricity(x), axis=1)
    return df


# def map_booleans(df):
#     mapping = {"yes": 1, "no": 0}
#     df['edjefa'] = df['edjefa'].replace(mapping).astype(np.float64)
#     df['edjefe'] = df['edjefe'].replace(mapping).astype(np.float64)
#     return df


def fix_outliers(df):
    # According to competition host (https://www.kaggle.com/c/costa-rican-household-poverty-prediction/discussion/61403)
    # we can safely change the rez_esc value to 5
    df.loc[df['rez_esc'] == 99.0, 'rez_esc'] = 5
    return df


def feature_engineer_rent(df):

    # Fill in households that fully own house with 0 rent payment
    df.loc[(df['tipovivi1'] == 1), 'v2a1'] = 0  # tipovivi1, =1 own and fully paid house
    df['v2a1_missing'] = np.where(df['v2a1'].isnull(), 1, 0)

    df['rent_by_hhsize'] = df['v2a1'] / df['hhsize']  # rent by household size
    df['rent_by_people'] = df['v2a1'] / df['r4t3']  # rent by people in household
    df['rent_by_rooms'] = df['v2a1'] / df['rooms']  # rent by number of rooms
    df['rent_per_bedroom'] = df['v2a1'] / df['bedrooms']
    df['rent_by_living'] = df['v2a1'] / df['tamviv']  # rent by number of persons living in the household
    df['rent_by_minor'] = df['v2a1'] / df['hogar_nin']
    df['rent_by_adult'] = df['v2a1'] / df['hogar_adul']
    df['rent_by_dep'] = df['v2a1'] / df['calc_dependency']
    df['rent_by_dep_count'] = df['v2a1'] / df['dependency_count']
    # df['rent_by_educ'] = df['v2a1'] / df['meaneduc']
    # df['rent_by_numPhone'] = df['v2a1'] / df['qmobilephone']
    # df['rent_by_gadgets'] = df['v2a1'] / (df['computer'] + df['mobilephone'] + df['v18q'])
    # df['rent_by_num_gadgets'] = df['v2a1'] / (df['v18q1'] + df['qmobilephone'])
    # df['rent_by_appliances'] = df['v2a1'] / df['appliances']

    #Top code at #1 million per whatever (mostly to avoid inf
    for col in ['rent_by_minor', 'rent_by_adult', 'rent_by_dep']:
        df.loc[df[col] > 1000000, col] = 1000000
    return df


def feature_engineer_education(df):
    # A few individuals (incl heads of hh) have missing meaneduc but nonmissing escolari
    df.loc[df['meaneduc'].isnull(), 'meaneduc'] = df['escolari']

    # rez_esc is not defined for certain ages, but let's add 0 to missing for later transformations
    df.loc[df['rez_esc'].isnull(), 'rez_esc'] = 0

    df['no_primary_education'] = df['instlevel1'] + df['instlevel2']

    # Different ways of scaling years behind in school to school years
    df['rez_esc_scaled'] = df['rez_esc'] / (df['age'] - 6)  # years behind per year (ages 7-17)
    df['rez_esc_escolari'] = df['rez_esc'] / df['escolari']

    # Aggregate some hh-level characteristics
    df['age_7_17'] = np.where((df['age'] >= 7) & (df['age'] <= 17), 1, 0)
    df['hh_max_rez_esc'] = df.groupby('idhogar')['rez_esc'].transform(lambda x: x.max())
    df['hh_sum_rez_esc'] = df.groupby('idhogar')['rez_esc'].transform(lambda x: x.sum())

    df['hh_max_rez_esc_scaled'] = df.groupby('idhogar')['rez_esc_scaled'].transform(lambda x: x.max())
    df['hh_sum_rez_esc_scaled'] = df.groupby('idhogar')['rez_esc_scaled'].transform(lambda x: x.sum())
    df['hh_children_7_17'] = df.groupby('idhogar')['age_7_17'].transform(lambda x: x.sum())

    df['hh_rez_esc_pp'] = df['hh_sum_rez_esc'] / df['hh_children_7_17']

    # drop intermediate vars
    df.drop(['hh_children_7_17', 'age_7_17'], axis=1, inplace=True)

    # For future: try as per https://www.kaggle.com/gaxxxx/exploratory-data-analysis-lightgbm
    # cols = ['edjefe', 'edjefa']
    # df[cols] = df[cols].replace({'no': 0, 'yes': 1}).astype(float)

    return df


def feature_engineer_age_composition(df):
    # Especially training on heads of hh, important to ID houses with a minor as head of household
    df['adult'] = np.where(df['age'] > 18, 1, 0)

    df['adult_minus_child'] = df['hogar_adul'] - df['hogar_mayor']
    df['child_percent'] = df['hogar_nin'] / df['hogar_total']
    df['elder_percent'] = df['hogar_mayor'] / df['hogar_total']
    df['adult_percent'] = df['hogar_adul'] / df['hogar_total']

    df['r4h1_percent_in_male'] = df['r4h1'] / df['r4h3']
    df['r4m1_percent_in_female'] = df['r4m1'] / df['r4m3']
    df['r4h1_percent_in_total'] = df['r4h1'] / df['hhsize']
    df['r4m1_percent_in_total'] = df['r4m1'] / df['hhsize']
    df['r4t1_percent_in_total'] = df['r4t1'] / df['hhsize']

    df['age_12_19'] = df['hogar_nin'] - df['r4t1']
    df['escolari_age'] = df['escolari'] / df['age']

    return df


def feature_engineer_housing_quality(df):
    df['wall_quality'] = 0 * df['epared1'] + 1 * df['epared2'] + 2 * df['epared3']
    df['roof_quality'] = 0 * df['etecho1'] + 1 * df['etecho2'] + 2 * df['etecho3']
    df['floor_quality'] = 0 * df['eviv1'] + 1 * df['eviv2'] + 2 * df['eviv3']
    df['house_material_vulnerability'] = df['epared1'] + df['etecho1'] + df['eviv1']

    return df

def feature_engineer_house_characteristics(df):
    df['dependency_count'] = df['hogar_nin'] + df['hogar_mayor']
    # Original 'dependency' variable has 'yes' and 'no', so better to calculate it per definition
    df['calc_dependency'] = df['dependency_count'] / df['hogar_adul']
    df.loc[
        df['hogar_adul'] == 0, 'calc_dependency'] = 8  # this is hacky, but original dependency var is 8 when no adults
    df['calc_dependency_bin'] = np.where(df['calc_dependency'] == 0, 0, 1)

    df['overcrowding_room_and_bedroom'] = (df['hacdor'] + df['hacapo']) / 2
    df['rooms_pc'] = df['rooms'] / df['tamviv']
    df['bedroom_per_room'] = df['bedrooms'] / df['rooms']
    df['elder_per_room'] = df['hogar_mayor'] / df['rooms']
    df['adults_per_room'] = df['adult'] / df['rooms']
    df['child_per_room'] = df['hogar_nin'] / df['rooms']
    df['male_per_room'] = df['r4h3'] / df['rooms']
    df['female_per_room'] = df['r4m3'] / df['rooms']
    df['room_per_person_household'] = df['hhsize'] / df['rooms']
    df['elder_per_bedroom'] = df['hogar_mayor'] / df['bedrooms']
    df['adults_per_bedroom'] = df['adult'] / df['bedrooms']
    df['child_per_bedroom'] = df['hogar_nin'] / df['bedrooms']
    df['male_per_bedroom'] = df['r4h3'] / df['bedrooms']
    df['female_per_bedroom'] = df['r4m3'] / df['bedrooms']
    df['bedrooms_per_person_household'] = df['hhsize'] / df['bedrooms']

    return df


def feature_engineer_assets(df):
    df.loc[df['v18q'] == 0, 'v18q1'] = 0  # num tablets = 0 if has_tablet == 0
    df['tablet_per_person_household'] = df['v18q1'] / df['hhsize']
    df['phone_per_person_household'] = df['qmobilephone'] / df['hhsize']

    # df['phones_pc'] = df['qmobilephone'] / df['tamviv']
    # df['tablets_pc'] = df['v18q1'] / df['tamviv']

    df['asset_index1'] = df['refrig'] + df['computer'] + df['television']
    df['asset_index2'] = df['asset_index1'] + df['v18q']

    df['house_utility_vulnerability'] = (df['sanitario1'] +
                                       df['pisonotiene'] +
                                       (df['cielorazo'] == 0) +  # coding is opposite (the house has ceiling)
                                       df['abastaguano'] +
                                       df['noelec'] +
                                       df['sanitario1'])

    return df


def run_feature_engineering(df):
    df = feature_engineer_education(df)
    df = feature_engineer_age_composition(df)
    df = feature_engineer_housing_quality(df)
    df = feature_engineer_house_characteristics(df)
    df = feature_engineer_rent(df)
    df = feature_engineer_assets(df)

    return df


def preprocess(df):

    df = clean_dummy_features(df)
    df = fix_outliers(df)

    df = run_feature_engineering(df)

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

