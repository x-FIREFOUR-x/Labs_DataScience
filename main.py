import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
desired_width = 320
pd.set_option('display.width', desired_width)


    #зчитати датасет в датафрейм
def read_dataset(path):
    data = pd.read_csv(path, sep=';', encoding='cp1252')
    return data


    #конвертувати тип даних колонки column_name датафрейма з string в float
def convert_column_str_to_float(dataset, column_name):
    dataset[column_name] = dataset[column_name]\
        .str.replace(',', '.')\
        .astype(float)


    #замінити NaN на середні значення
def replace_nan_to_avarege(dataset):
    return dataset.fillna(dataset.mean())


    #вивести негативні елементи колонок
def print_neg_elements_dataset(dataset):
    print('GDP per capita negative values:\n', dataset[dataset['GDP per capita'] < 0])
    print('\nPopulation negative values:\n', dataset[dataset['Populatiion'] < 0])
    print('\nCO2 emission negative values:\n', dataset[dataset['CO2 emission'] < 0])
    print('\nArea negative values:\n', dataset[dataset['Area'] < 0])


    #замінити від'ємні значення в колонці column_name на їх абсолютні значення
def abs_date(dataset, column_name):
     dataset[column_name] = dataset[column_name].abs()



if __name__ == '__main__':
    data_path = 'data\Data2.csv'
    dataset = read_dataset(data_path)

    convert_column_str_to_float(dataset, 'GDP per capita')
    convert_column_str_to_float(dataset, 'CO2 emission')
    convert_column_str_to_float(dataset, 'Area')

    dataset = replace_nan_to_avarege(dataset)

    # print_neg_elements_dataset(dataset)

    abs_date(dataset, 'GDP per capita')
    abs_date(dataset, 'Area')

    #print(dataset.head(217))
    #print(dataset.info())