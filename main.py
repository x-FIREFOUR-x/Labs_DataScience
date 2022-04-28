import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
desired_width = 320
pd.set_option('display.width', desired_width)

    #зчитати дасет в датафрейм
def read_dataset(path):
    data = pd.read_csv(path, sep=';', encoding='cp1252')
    return data

    #вивести датасет
def print_dataset(dataset):
   print(dataset)

    #конвертувати тип даних колонки column_name датафрейма з string в float
def convert_column_str_to_float(dataset, column_name):
    dataset[column_name] = dataset[column_name]\
        .str.replace(',', '.')\
        .astype(float)

    #замінити NaN на середні значення
def replace_nan_to_avarege(dataset):
    return dataset.fillna(dataset.mean())

    #замінити від'ємні значення в колонці column_name на їх абсолютні значення
def abs_date(dataset, column_name):
    dataset[column_name] = dataset[column_name].abs()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_path = 'data\Data2.csv'
    dataset = read_dataset(data_path)

    convert_column_str_to_float(dataset, 'GDP per capita')
    convert_column_str_to_float(dataset, 'CO2 emission')
    convert_column_str_to_float(dataset, 'Area')

    dataset = replace_nan_to_avarege(dataset)

    abs_date(dataset, 'GDP per capita')
    abs_date(dataset, 'Populatiion')
    abs_date(dataset, 'CO2 emission')
    abs_date(dataset, 'Area')

    print_dataset(dataset)