import pandas as pd

def read_jhu_conf(file_loc = '../data/time_series_covid19_confirmed_US.csv'):
    """
    Reads the JHU Confirmed Case Count Data.
    :param file_loc: file location
    :return: DataFrame of confirmed case counts
    """
    return None

def read_jhu_death(file_loc = '../data/time_series_covid19_deaths_US.csv'):
    """
    Read the JHU Death Count Data.
    :param file_loc: file location
    :return: DataFrame of death cases
    """
    return None

def read_cdph_test(file_loc = '../data/tbl2_testing_agg_March_12_2021.xlsx'):
    test_data = pd.read_excel(file_loc)
    test_data['county'] = test_data['county'].str.lower()
    test_data = test_data.dropna(subset=['county'])
    test_data['county'].str.split(',')
    test_data['county'] = [c[0] for c in test_data['county'].str.split(',')]
    return None