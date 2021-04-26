import pandas as pd

def read_jhu_conf(file_loc = '../data/time_series_covid19_confirmed_US.csv'):
    """
    Reads the JHU Confirmed Case Count Data.
    :param file_loc: file location
    :return: DataFrame of confirmed case counts
    """
    confirmed = pd.read_csv(file_loc)
    date_cols = confirmed.columns[11:]
    
    return None

def read_jhu_death(file_loc = '../data/time_series_covid19_deaths_US.csv'):
    """
    Read the JHU Death Count Data.
    :param file_loc: file location
    :return: DataFrame of death cases
    """
    return None

def read_cdph_test(file_loc = '../data/CDPH_testing_data_4_24.xlsx',
                   county_name_file = '../data/county_names.csv'):
    test_data = pd.read_excel(file_loc)
    test_data['county'] = test_data['county'].str.lower()
    test_data = test_data.dropna(subset=['county'])
    test_data['county'].str.split(',')
    test_data['county'] = [c[0] for c in test_data['county'].str.split(',')]
    county_names = pd.read_csv(county_name_file, dtype=str)
    county_names['County'] = county_names['County'].str.lower()
    county_names = county_names.set_index('County')
    test_data = test_data.join(county_names,on='county')
    return test_data


