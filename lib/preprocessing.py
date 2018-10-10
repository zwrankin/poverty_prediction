import pandas as pd
import numpy as np

# IMPORTANT: according to the competition (https://www.kaggle.com/c/costa-rican-household-poverty-prediction/discussion/61403),
# ONLY heads of household are used in scoring, so we should only train on them (with some household aggregate features)
SUBSET_TO_HEAD_OF_HOUSEHOLD = False

# Next three functions are from https://www.kaggle.com/gaxxxx/exploratory-data-analysis-lightgbm
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


def name_correlated_feature_to_drop(df, threshold=0.90):
    """
    Returns a list of HALF the correlated features
    :param df: pd.DataFrame
    :param threshold: threshold at which one correlated feature will be returned
    :return: list of variables
    """
    corr_matrix = df.corr()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation >= threshold
    to_drop = [column for column in upper.columns if any(abs(upper[column]) >= threshold)]

    return to_drop


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
        varlist += vars_level2
        varlist += vars_level3

    return varlist


def feature_engineer_rent(df, level='low', drop_correlated_features=False):

    varlist = select_varlist(level,
                             vars_level1=['v2a1', 'tipovivi_rank', 'rent_by_dep', 'tipovivi4', 'tipovivi2'],
                             vars_level2=['rent_by_people', 'rent_by_rooms', 'tipovivi5', 'tipovivi1', 'tipovivi3'],
                             vars_level3=['rent_by_hhsize', 'rent_per_bedroom',
                                          'rent_by_living', 'rent_by_minor', 'rent_by_adult',
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

    df['tipovivi_rank'] = np.NaN
    df['tipovivi_rank'] = 0 * df['tipovivi4'] + df['tipovivi5'] + 2 * df['tipovivi1'] + 3 * df['tipovivi3'] + \
                      4 * df['tipovivi2']

    # Top code at #1 million per whatever (mostly to avoid inf)
    for col in ['rent_by_minor', 'rent_by_adult', 'rent_by_dep', 'rent_by_dep_count']:
        df.loc[df[col] > 1000000, col] = 1000000

    if drop_correlated_features:
        # to_drop = name_correlated_feature_to_drop(df)
        to_drop = ['tipovivi4']
        varlist = [v for v in varlist if v not in to_drop]

    if SUBSET_TO_HEAD_OF_HOUSEHOLD:
        df = df.query('parentesco1 == 1')

    return df[varlist]


def feature_engineer_education(df, level='low', drop_correlated_features=False):

    varlist = select_varlist(level,
                             vars_level1=['meaneduc', 'escolari', 'educ_rank', 'rez_esc',
                                          'no_primary_education', 'higher_education'],
                             vars_level2=['rez_esc_escolari',
                                          'instlevel2', 'instlevel1', 'instlevel9', 'instlevel8'],
                             vars_level3=['instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7'])


    # A few individuals (incl heads of hh) have missing meaneduc but nonmissing escolari
    df.loc[df['meaneduc'].isnull(), 'meaneduc'] = df['escolari']

    # rez_esc is not defined for certain ages, but let's add 0 to missing for later transformations
    df.loc[df['rez_esc'].isnull(), 'rez_esc'] = 0

    df['no_primary_education'] = df['instlevel1'] + df['instlevel2']
    df['higher_education'] = df['instlevel8'] + df['instlevel9']

    df['educ_rank'] = np.NaN
    df['educ_rank'] = 0 * df['instlevel2'] + df['instlevel1'] + 2 * df['instlevel3'] + 3 * df['instlevel4'] + \
                      4 * df['instlevel6'] + 5 * df['instlevel7'] + 6 * df['instlevel5'] + 7 * df['instlevel9'] + \
                      8 * df['instlevel8']

    # Different ways of scaling years behind in school to school years
    df['rez_esc_scaled'] = df['rez_esc'] / (df['age'] - 6)  # years behind per year (ages 7-17)
    df['rez_esc_escolari'] = df['rez_esc'] / df['escolari']
    df.loc[df['escolari'] == 0, 'rez_esc_escolari'] = 5  # top code (for when escolari = 0)

    df['age_7_17'] = np.where((df['age'] >= 7) & (df['age'] <= 17), 1, 0)

    # agg_varlist = ['meaneduc', 'rez_esc', 'rez_esc_scaled', 'age_7_17']
    # df = aggregate_features(df, agg_varlist, ['mean', 'sum', 'max'])

    # df['hh_rez_esc_pp'] = df['rez_esc_sum'] / df['age_7_17_sum']

    if drop_correlated_features:
        # to_drop = name_correlated_feature_to_drop(df)
        to_drop = ['escolari', 'educ_rank', 'no_primary_education', 'higher_education']
        varlist = [v for v in varlist if v not in to_drop]

    if SUBSET_TO_HEAD_OF_HOUSEHOLD:
        df = df.query('parentesco1 == 1')

    return df[varlist]


def feature_engineer_demographics(df, level='low', drop_correlated_features=False):
    varlist = select_varlist(level,
                             vars_level1=['adult', 'adult_percent', 'child_percent', 'r4t1_percent_in_total',
                                          'tamviv', 'hhsize', 'estadocivil3', 'estadocivil1', 'hogar_nin', 'hogar_adul',
                                          'lugar1', 'area1', 'age'],
                             vars_level2=['dis', 'male', 'female', 'adult_minus_child', 'elder_percent',
                                          'dis_sum', 'hogar_mayor', 'hogar_total',
                                          'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6'],
                             vars_level3=['r4h1_percent_in_male', 'r4m1_percent_in_female', 'r4h1_percent_in_total',
                                          'r4m1_percent_in_total', 'age_12_19', 'escolari_age', 'tamhog',
                                          'dis_mean', 'adult_mean', 'male_mean', 'female_mean',
                                          'dis_sum', 'adult_sum', 'male_sum', 'female_sum'],
                             )

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

    agg_varlist = ['dis', 'adult', 'male', 'female']
    df = aggregate_features(df, agg_varlist, ['mean', 'sum'])

    if drop_correlated_features:
        # to_drop = name_correlated_feature_to_drop(df)
        to_drop = ['child_percent', 'r4t1_percent_in_total', 'hhsize', 'hogar_nin', 'age']
        varlist = [v for v in varlist if v not in to_drop]

    if SUBSET_TO_HEAD_OF_HOUSEHOLD:
        df = df.query('parentesco1 == 1')

    return df[varlist]


def feature_engineer_house_characteristics(df, level='low', drop_correlated_features=False):
    varlist = select_varlist(level,
                             vars_level1=['dependency_count', 'calc_dependency',
                                          'child_per_bedroom', 'female_per_bedroom', 'male_per_bedroom',
                                          'bedrooms', 'rooms', 'bedrooms_per_person_household'],
                             vars_level2=['hacdor', 'hacapo', 'calc_dependency_bin', 'rooms_pc',
                                          'child_per_room', 'male_per_room', 'female_per_room',
                                          'overcrowding_room_and_bedroom'],
                             vars_level3=['bedroom_per_room', 'elder_per_room', 'adults_per_room',
                                          'room_per_person_household', 'elder_per_bedroom', 'adults_per_bedroom',
                                          ])

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

    if drop_correlated_features:
        # to_drop = name_correlated_feature_to_drop(df)
        to_drop = ['calc_dependency', 'rooms', 'bedrooms_per_person_household']
        varlist = [v for v in varlist if v not in to_drop]

    if SUBSET_TO_HEAD_OF_HOUSEHOLD:
        df = df.query('parentesco1 == 1')

    return df[varlist]


def feature_engineer_house_rankings(df, level='low', drop_correlated_features=False):
    # Note - I'm making the arbitrary decision that any individual aspect with >10% correlation and n > 100 is medium
    # and any with >20% and n > 100 is level1
    # Note - noelec is level2 because n = 21
    varlist = select_varlist(level,
                             vars_level1=['floor_rank', 'wall_rank', 'roof_rank', 'water_rank', 'electricity_rank',
                                          'toilet_rank', 'cooking_rank', 'trash_rank',
                                          'wall_quality', 'roof_quality', 'floor_quality',
                                          'house_material_bad', 'house_material_good',
                                          'material_rank_sum', 'utility_rank_sum',
                                          'cielorazo', 'energcocinar4', 'pisocemento', 'pisomoscer', 'paredblolad',
                                          'epared1', 'epared3', 'etecho1', 'etecho3', 'eviv1', 'eviv3'],
                             vars_level2=['pisonotiene', 'paredmad', 'abastaguadentro',
                                          'noelec', 'energcocinar2',
                                          'elimbasu3', 'elimbasu1', 'epared2', 'etecho2', 'eviv2'],
                             vars_level3=['pisoother', 'pisonatur', 'pisomadera', 'public', 'sanitario5',
                                          'paredpreb', 'pareddes', 'paredzocalo', 'paredzinc', 'paredfibras', 'paredother',
                                          'techocane', 'techoentrepiso', 'techozinc', 'techootro',
                                          'abastaguafuera', 'abastaguano', 'coopele',
                                          'sanitario1', 'sanitario2', 'sanitario3', 'sanitario6',
                                          'energcocinar3', 'energcocinar1',
                                          'elimbasu2', 'elimbasu4', 'elimbasu6',
                                          ])

    df['floor_rank'] = np.NaN
    df['floor_rank'] = 0 * df ['pisonotiene'] + df['pisocemento'] + 2 * df['pisomadera'] + 3 * df['pisonatur'] + \
                       4 * df['pisoother'] + 5 * df['pisomoscer']

    df['wall_rank'] = np.NaN
    df['wall_rank'] = 0 * df['paredmad'] + df['paredpreb'] + 2 * df['pareddes'] + 3 * df['paredzocalo'] + \
                      4 * df['paredzinc'] + 5 * df['paredfibras'] + 6 * df['paredother'] + 7 * df['paredblolad']

    df['roof_rank'] = np.NaN
    df['roof_rank'] = 0 * df['techocane'] + df['techoentrepiso'] + 2 * df['techozinc'] + 3 * df['techootro']

    df['water_rank'] = np.NaN
    df['water_rank'] = 0 * df['abastaguano'] + df['abastaguafuera'] + 2 * df['abastaguadentro']

    df['electricity_rank'] = np.NaN
    df['electricity_rank'] = 0 * df['noelec'] + df['planpri'] + 2 * df['coopele'] + 3 * df['public']

    # ranking off correlations even though they don't logically align (e.g., no toilet isn't the worst)
    df['toilet_rank'] = np.NaN
    df['toilet_rank'] = 0 * df['sanitario5'] + df['sanitario1'] + 2 * df['sanitario3'] + \
                        3 * df['sanitario6'] + 4 * df['sanitario2']

    df['cooking_rank'] = np.NaN
    df['cooking_rank'] = 0 * df['energcocinar4'] + df['energcocinar3'] + 2 * df['energcocinar1'] + 3 * df['energcocinar2']

    df['trash_rank'] = np.NaN
    df['trash_rank'] = 0 * df['elimbasu3'] + df['elimbasu2'] + 2 * df['elimbasu4'] + 3 * df['elimbasu6'] + \
                      4 * df['elimbasu1']

    df['wall_quality'] = np.NaN
    df['wall_quality'] = 0 * df['epared1'] + 1 * df['epared2'] + 2 * df['epared3']

    df['roof_quality'] = np.NaN
    df['roof_quality'] = 0 * df['etecho1'] + 1 * df['etecho2'] + 2 * df['etecho3']

    df['floor_quality'] = np.NaN
    df['floor_quality'] = 0 * df['eviv1'] + 1 * df['eviv2'] + 2 * df['eviv3']

    df['house_material_bad'] = df['epared1'] + df['etecho1'] + df['eviv1']
    df['house_material_good'] = df['epared3'] + df['etecho3'] + df['eviv3']

    df['material_rank_sum'] = np.NaN
    df['material_rank_sum'] = df['floor_rank'] + df['wall_rank'] + df['roof_rank']

    df['utility_rank_sum'] = np.NaN
    df['utility_rank_sum'] = df['water_rank'] + df['electricity_rank'] + df['toilet_rank'] + df['cooking_rank'] + df['trash_rank']

    if drop_correlated_features:
        # to_drop = name_correlated_feature_to_drop(df)
        to_drop = ['roof_quality', 'floor_quality', 'house_material_bad', 'house_material_good', 'material_rank_sum', 'cielorazo', 'pisocemento', 'pisomoscer', 'paredblolad', 'epared1', 'epared3', 'etecho1', 'etecho3', 'eviv1', 'eviv3']
        varlist = [v for v in varlist if v not in to_drop]

    if SUBSET_TO_HEAD_OF_HOUSEHOLD:
        df = df.query('parentesco1 == 1')

    return df[varlist]


def feature_engineer_assets(df, level='low', drop_correlated_features=False):
    varlist = select_varlist(level,
                             vars_level1=['v18q', 'asset_index', 'house_utility_vulnerability',
                                          'phone_per_person_household', ''
                                          'v14a', 'refrig', 'sanitario1',
                                          'computer', 'television', 'tech'],
                             vars_level2=['tablet_per_person_household', 'v18q1',
                                          'mobilephone', 'qmobilephone'],
                             vars_level3=[])

    df.loc[df['v18q'] == 0, 'v18q1'] = 0  # num tablets = 0 if has_tablet == 0
    df['tablet_per_person_household'] = df['v18q1'] / df['hhsize']
    df['phone_per_person_household'] = df['qmobilephone'] / df['hhsize']

    df['tech'] = np.NaN
    df['tech'] = df['v18q'] + df['mobilephone']

    df['asset_index'] = np.NaN
    df['asset_index'] = df['refrig'] + df['computer'] + df['television'] + df['mobilephone'] + df['v18q']

    df['house_utility_vulnerability'] = np.NaN
    df['house_utility_vulnerability'] = (df['sanitario1'] +
                                       df['pisonotiene'] +
                                       (df['cielorazo'] == 0) +  # coding is opposite (the house has ceiling)
                                       df['abastaguano'] +
                                       df['noelec'] +
                                       df['sanitario1'])

    if drop_correlated_features:
        # to_drop = name_correlated_feature_to_drop(df)
        to_drop = ['sanitario1']
        varlist = [v for v in varlist if v not in to_drop]

    if SUBSET_TO_HEAD_OF_HOUSEHOLD:
        df = df.query('parentesco1 == 1')

    return df[varlist]


def feature_engineer_aggregate_individuals(df):
    ind_bool = ['v18q', 'adult', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3',
                'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7',
                'parentesco1', 'parentesco2', 'parentesco3', 'parentesco4', 'parentesco5',
                'parentesco6', 'parentesco7', 'parentesco8', 'parentesco9', 'parentesco10',
                'parentesco11', 'parentesco12', 'instlevel1', 'instlevel2', 'instlevel3',
                'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8',
                'instlevel9', 'mobilephone', 'rez_esc-missing']

    ind_ordered = ['rez_esc', 'escolari', 'age']
    ind_engineered = ['educ_rank', 'rez_esc_scaled',
                      'no_primary_education', 'higher_education']
    ind_all = ind_bool + ind_ordered + ind_engineered

    cols = [col for col in ind_all if col in df.columns]

    functions = ['min', 'max', 'sum', 'count', 'std']
    agg = aggregate_features(df, cols, functions, idvar='idhogar').drop('idhogar', axis=1)

    # if later we just want newly created vars
    # keep_cols = []
    # for c in cols:
    #     for f in functions:
    #         keep_cols += [c + '_' + f]
    return agg


def basic_feature_engineering(df):
    """Create important features that are used by other feature engineering transformers"""
    df['dependency_count'] = df['hogar_nin'] + df['hogar_mayor']
    # Original 'dependency' variable has 'yes' and 'no', so better to calculate it per definition
    df['calc_dependency'] = df['dependency_count'] / df['hogar_adul']
    df.loc[
        df['hogar_adul'] == 0, 'calc_dependency'] = 8  # this is hacky, but original dependency var is 8 when no adults
    df['calc_dependency_bin'] = np.where(df['calc_dependency'] == 0, 0, 1)
    return df


def run_feature_engineering(df, level='low', drop_correlation_threshold=None):
    """
    Convenience function for running all feature engineering outside of pipeline
    :param df: pd.DataFrame
    :param level: level of feature engineering to be applied to ALL functions
    :param drop_correlation_threshold: if not None, then the threshold above which one of each pair of correlated
    featrues will be dropped
    :return: pd.DataFrame with only specified features
    """
    hh_ids = df['idhogar']
    df = pd.concat([feature_engineer_rent(df, level),
                    feature_engineer_education(df, level),
                    feature_engineer_demographics(df, level),
                    feature_engineer_house_rankings(df, level),
                    feature_engineer_house_characteristics(df, level),
                    feature_engineer_assets(df, level)],
                   axis=1)
    df['idhogar'] = hh_ids
    df = feature_engineer_aggregate_individuals(df)

    if drop_correlation_threshold:
        to_drop = name_correlated_feature_to_drop(df, threshold=drop_correlation_threshold)
        df.drop(to_drop, axis=1, inplace=True)

    return df


def preprocess(df):
    """Convenience function for data processing before pipeline"""
    df = clean_dummy_features(df)
    df = fix_outliers(df)
    df = basic_feature_engineering(df)

    return df


