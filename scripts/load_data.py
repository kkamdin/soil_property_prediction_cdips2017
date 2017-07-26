import numpy as np
import pandas as pd

DEFAULT_DATA_DIR='../data'
DEFAULT_TRAINING_DATA=DEFAULT_DATA_DIR + '/' + 'training.csv'
DEFAULT_TEST_DATA=DEFAULT_DATA_DIR + '/' + 'sorted_test.csv'


def load_training_spectra(data_path=DEFAULT_TRAINING_DATA, include_depth=False, include_pidn=False):
    """ Return the wavenumer colnums of the training set with the m stripped (with option to inlude sample depth and pidn) as a
        pandas DataFrame. Return the targets as well, but in a separate pandas DataFrame""" 
    #get all the training data
    data = pd.read_csv(data_path)
    #make a list of the wavenumber columns
    column_list = [column for column in data.columns if column.startswith('m')]
    #strip the 'm' from the wavenumbers
    wavenum_list = [column.lstrip('m') for column in column_list]
    #rename the wavenumber colnums, getting rid of the 'm' 
    data.rename(columns=dict(zip(column_list,wavenum_list)), inplace=True)
    columns_to_return = wavenum_list

    if include_depth:
        columns_to_return.append('Depth')

    if include_pidn:
        columns_to_return = ['PIDN'] + columns_to_return

    targets = ['Ca','P','pH','SOC','Sand']
    return data[columns_to_return], data[targets]


def load_training_satellite_data(data_path=DEFAULT_TRAINING_DATA, include_depth=False, include_pidn=False):
    """ Return the satellite data colnums of the training set (with options to inlude sample depth and pidn) as a
        pandas DataFrame. Return the targets as well, but in a separate pandas DataFrame""" 
    #get all the training data
    data = pd.read_csv(data_path)
    targets = ['Ca','P','pH','SOC','Sand']
    #make a list of the not-wavenumber column and remove the targetss
    column_list = [column for column in data.columns if not column.startswith('m')]
    columns_to_return = [col for col in column_list if col not in targets]
    if not include_depth:
        columns_to_return.remove('Depth')

    if not include_pidn:
        columns_to_return.remove('PIDN')

    return data[columns_to_return], data[targets]



def load_all_training_data(data_path=DEFAULT_TRAINING_DATA, include_pidn=False):
    """ Return the all the data colnums of the training set (with option to include pidn) as a
        pandas DataFrame. Wavenumbers are stripped of the leading 'm'. Return the targets as well,
         but in a separate pandas DataFrame""" 
    #get all the training data
    data = pd.read_csv(data_path)
    targets = ['Ca','P','pH','SOC','Sand']
    #make a list of the not-wavenumber column and remove the targetss
    temp_list = [column for column in data.columns if not column.startswith('m')]
    satellite_list = [col for col in temp_list if col not in targets] #this includes Depth and PIDN

    old_wavenums = [column for column in data.columns if column.startswith('m')]
    #strip the 'm' from the wavenumbers
    new_wavenums = [column.lstrip('m') for column in old_wavenums]
    data.rename(columns=dict(zip(old_wavenums,new_wavenums)), inplace=True)

    columns_to_return = new_wavenums + satellite_list    

    if not include_pidn:
        columns_to_return.remove('PIDN')

    return data[columns_to_return], data[targets]


def load_test_spectra(data_path=DEFAULT_TEST_DATA, include_depth=False, include_pidn=False):
    """ Return the wavenumer colnums of the test set with the m stripped (with option to inlude sample depth and pidn) as a
        pandas DataFrame. Return the targets as well, but in a separate pandas DataFrame""" 
    #get all the test data
    data = pd.read_csv(data_path)
    print(data.head())
    #make a list of the wavenumber columns
    column_list = [column for column in data.columns if column.startswith('m')]
    #strip the 'm' from the wavenumbers
    wavenum_list = [column.lstrip('m') for column in column_list]
    #rename the wavenumber colnums, getting rid of the 'm' 
    data.rename(columns=dict(zip(column_list,wavenum_list)), inplace=True)
    columns_to_return = wavenum_list

    if include_depth:
        columns_to_return.append('Depth')

    if include_pidn:
        columns_to_return = ['PIDN'] + columns_to_return

    return data[columns_to_return]


def load_test_satellite_data(data_path=DEFAULT_TEST_DATA, include_depth=False, include_pidn=False):
    """ Return the satellite data colnums of the test set (with options to inlude sample depth and pidn) as a
        pandas DataFrame. Return the targets as well, but in a separate pandas DataFrame""" 
    #get all the test data
    data = pd.read_csv(data_path)
    #make a list of the not-wavenumber column and remove the targetss
    column_list = [column for column in data.columns if not column.startswith('m')]
    columns_to_return = column_list
    if not include_depth:
        columns_to_return.remove('Depth')

    if not include_pidn:
        columns_to_return.remove('PIDN')

    return data[columns_to_return]



def load_all_test_data(data_path=DEFAULT_TEST_DATA, include_pidn=False):
    """ Return the all the data colnums of the test set (with option to include pidn) as a
        pandas DataFrame. Wavenumbers are stripped of the leading 'm'. Return the targets as well,
         but in a separate pandas DataFrame""" 
    #get all the test data
    data = pd.read_csv(data_path)
    #make a list of the not-wavenumber column and remove the targetss
    temp_list = [column for column in data.columns if not column.startswith('m')]
    satellite_list = temp_list #this includes Depth and PIDN

    old_wavenums = [column for column in data.columns if column.startswith('m')]
    #strip the 'm' from the wavenumbers
    new_wavenums = [column.lstrip('m') for column in old_wavenums]
    data.rename(columns=dict(zip(old_wavenums,new_wavenums)), inplace=True)

    columns_to_return = new_wavenums + satellite_list    

    if not include_pidn:
        columns_to_return.remove('PIDN')

    return data[columns_to_return]


