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


def aggregate_features(df, varlist, functions, idvar='idhogar'):
    """
    Function to aggregate variables of interest as new variables in existing dataframe
    :param df: pd.DataFrame
    :param varlist: str or list of variables to aggregate
    :param functions: str or list of agg functions to apply on varlist
    :param idvar: household identifier
    :return: pd.DataFrame df with additional columns merged on
    """

    if isinstance(varlist, str):
        varlist = [varlist]

    if isinstance(functions, str):
        functions = [functions]

    for f in functions:
        varlist2 = [f'{var}_{f}' for var in varlist]
        df2 = df.groupby(idvar)[varlist].agg(f)
        df2.columns = varlist2
        df = pd.merge(df, df2.reset_index(), on='idhogar')

    return df


def select_varlist(level: str, vars_level1: list, vars_level2: list, vars_level3: list):
    """
    Convenience function to return the appropriate cumulative varlist for a specified level of detail
    :param level: str, one of ['low', 'medium', 'high']
    :param vars_level1: base varlist
    :param vars_level2: varlist to be added if level == 'medium'
    :param vars_level3: varlist to be added if level == 'high'
    """

    if level not in ['low', 'medium', 'high']:
        raise AssertionError(f'{level} not an acceptable level')

    varlist = vars_level1

    if level == 'medium':
        varlist += vars_level2
    elif level == 'high':
        varlist += vars_level3

    return varlist


def feature_engineer_rent(df, level='low'):

    varlist = select_varlist(level,
                             vars_level1=['v2a1'],
                             vars_level2=['rent_by_dep', 'rent_by_dep_count'],
                             vars_level3=['rent_by_hhsize', 'rent_by_people', 'rent_by_rooms', 'rent_per_bedroom',
                                          'rent_by_living', 'rent_by_minor', 'rent_by_adult', 'rent_by_dep',
                                          'rent_by_dep_count'])

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

    # Top code at #1 million per whatever (mostly to avoid inf)
    for col in ['rent_by_minor', 'rent_by_adult', 'rent_by_dep', 'rent_by_dep_count']:
        df.loc[df[col] > 1000000, col] = 1000000

    return df[varlist]


def feature_engineer_education(df, level='low'):

    varlist = select_varlist(level,
                             vars_level1=['meaneduc', 'escolari', 'rez_esc', 'rez_esc_scaled'],
                             vars_level2=['no_primary_education', 'rez_esc_escolari',
                                          'meaneduc_mean', 'rez_esc_mean', 'rez_esc_scaled_mean'],
                             vars_level3=['meaneduc_sum', 'rez_esc_sum', 'rez_esc_scaled_sum'])


    # A few individuals (incl heads of hh) have missing meaneduc but nonmissing escolari
    df.loc[df['meaneduc'].isnull(), 'meaneduc'] = df['escolari']

    # rez_esc is not defined for certain ages, but let's add 0 to missing for later transformations
    df.loc[df['rez_esc'].isnull(), 'rez_esc'] = 0

    df['no_primary_education'] = df['instlevel1'] + df['instlevel2']

    # Different ways of scaling years behind in school to school years
    df['rez_esc_scaled'] = df['rez_esc'] / (df['age'] - 6)  # years behind per year (ages 7-17)
    df['rez_esc_escolari'] = df['rez_esc'] / df['escolari']
    df.loc[df['escolari'] == 0, 'rez_esc_escolari'] = 5  # top code (for when escolari = 0)

    df['age_7_17'] = np.where((df['age'] >= 7) & (df['age'] <= 17), 1, 0)

    agg_varlist = ['meaneduc', 'rez_esc', 'rez_esc_scaled', 'age_7_17']
    df = aggregate_features(df, agg_varlist, ['mean', 'sum', 'max'])

    df['hh_rez_esc_pp'] = df['rez_esc_sum'] / df['age_7_17_sum']

    return df[varlist]


def feature_engineer_age_composition(df, level='low'):
    varlist = select_varlist(level,
                             vars_level1=['adult'],
                             vars_level2=['adult_minus_child', 'child_percent', 'elder_percent', 'adult_percent'],
                             vars_level3=['r4h1_percent_in_male', 'r4m1_percent_in_female', 'r4h1_percent_in_total',
                                          'r4m1_percent_in_total', 'r4t1_percent_in_total', 'age_12_19', 'escolari_age'])

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

    return df[varlist]


def feature_engineer_housing_quality(df, level='low'):
    varlist = select_varlist(level,
                             vars_level1=['wall_quality', 'roof_quality', 'floor_quality',
                                          'house_material_vulnerability'],
                             vars_level2=['epared1', 'etecho1', 'eviv1'],
                             vars_level3=[])

    df['wall_quality'] = 0 * df['epared1'] + 1 * df['epared2'] + 2 * df['epared3']
    df['roof_quality'] = 0 * df['etecho1'] + 1 * df['etecho2'] + 2 * df['etecho3']
    df['floor_quality'] = 0 * df['eviv1'] + 1 * df['eviv2'] + 2 * df['eviv3']
    df['house_material_vulnerability'] = df['epared1'] + df['etecho1'] + df['eviv1']

    return df[varlist]


def feature_engineer_house_characteristics(df, level='low'):
    varlist = select_varlist(level,
                             vars_level1=['dependency_count', 'calc_dependency'],
                             vars_level2=['calc_dependency_bin', 'overcrowding_room_and_bedroom', 'rooms_pc'],
                             vars_level3=['bedroom_per_room', 'elder_per_room', 'adults_per_room',
                                          'child_per_room', 'male_per_room', 'female_per_room',
                                          'room_per_person_household', 'elder_per_bedroom', 'adults_per_bedroom',
                                          'child_per_bedroom', 'male_per_bedroom', 'female_per_bedroom',
                                          'bedrooms_per_person_household'])

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

    return df[varlist]


def feature_engineer_assets(df, level='low'):
    varlist = select_varlist(level,
                             vars_level1=['v18q', 'v18q1', 'asset_index1', 'house_utility_vulnerability'],
                             vars_level2=['tablet_per_person_household', 'phone_per_person_household', 'asset_index2'],
                             vars_level3=[])

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

    return df[varlist]

#
# def feature_engineer_household_aggregates(df, idvar='idhogar'):
#
#     varlist_mean = ['rez_esc', 'dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3',
#                       'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 'parentesco2',
#                       'parentesco3', 'parentesco4', 'parentesco5', 'parentesco6', 'parentesco7', 'parentesco8',
#                       'parentesco9', 'parentesco10', 'parentesco11', 'parentesco12',
#                       'instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7',
#                       'instlevel8', 'instlevel9', ]
#     varlist_mean2 = [f'{var}_mean' for var in varlist_mean]
#
#     df2 = df.groupby(idvar)[varlist_mean].mean()
#     df2.columns = varlist_mean2
#
#     varlist_all = ['escolari', 'age', 'escolari_age']
#
#     for function in ['mean', 'std', 'min', 'max', 'sum']:
#         hh_agg = df.groupby(idvar)[varlist_all].agg(function)
#         varlist_all2 = [f'{var}_{function}' for var in varlist_all]
#         hh_agg.columns = varlist_all2
#         df2 = df2.merge(hh_agg, left_index=True, right_index=True)
#
#     return pd.merge(df, df2.reset_index(), on='idhogar')


def basic_feature_engineering(df):
    """Create important features that are used by other feature engineering transformers"""
    df['dependency_count'] = df['hogar_nin'] + df['hogar_mayor']
    # Original 'dependency' variable has 'yes' and 'no', so better to calculate it per definition
    df['calc_dependency'] = df['dependency_count'] / df['hogar_adul']
    df.loc[
        df['hogar_adul'] == 0, 'calc_dependency'] = 8  # this is hacky, but original dependency var is 8 when no adults
    df['calc_dependency_bin'] = np.where(df['calc_dependency'] == 0, 0, 1)
    return df


def run_feature_engineering(df, level='low'):
    """
    Convenience function for running all feature engineering outside of pipeline
    :param df: pd.DataFrame
    :param level: level of feature engineering to be applied to ALL functions
    :return: pd.DataFrame with only specified features
    """
    df = pd.concat([feature_engineer_rent(df, level),
                    feature_engineer_education(df, level),
                    feature_engineer_age_composition(df, level),
                    feature_engineer_housing_quality(df, level),
                    feature_engineer_house_characteristics(df, level),
                    feature_engineer_assets(df, level)],
                   axis=1)

    return df


def preprocess(df):

    df = clean_dummy_features(df)
    df = fix_outliers(df)

    df = basic_feature_engineering(df)
    # df = run_feature_engineering(df)


    # Drop object columns that are not necessary for model
    # df.drop(['Id', 'idhogar', 'dependency', 'edjefe', 'edjefa'], axis=1, inplace=True)

    # # Remove useless feature to reduce dimension
    # train.drop(columns=['idhogar', 'Id', 'tamhog', 'agesq', 'hogar_adul', 'SQBescolari', 'SQBage', 'SQBhogar_total',
    #                     'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned'], inplace=True)
    # test.drop(columns=['idhogar', 'Id', 'tamhog', 'agesq', 'hogar_adul', 'SQBescolari', 'SQBage', 'SQBhogar_total',
    #                    'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned'], inplace=True)

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

