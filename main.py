import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
desired_width = 320
pd.set_option('display.width', desired_width)

def read_dataset(path):
    data = pd.read_csv(path, sep=';', encoding='cp1252')
    return data


def print_dataset(dataset):
   print(dataset)

def convert_column_str_to_float(dataset, column_name):
    dataset[column_name] = dataset[column_name]\
        .str.replace(',', '.')\
        .astype(float)

def replace_nan_to_avarege(dataset):
    return dataset.fillna(dataset.mean())

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_path = 'data\Data2.csv'
    dataset = read_dataset(data_path)

    convert_column_str_to_float(dataset, 'GDP per capita')
    convert_column_str_to_float(dataset, 'CO2 emission')
    convert_column_str_to_float(dataset, 'Area')

    print(dataset.info())

    dataset = replace_nan_to_avarege(dataset)

    print_dataset(dataset)