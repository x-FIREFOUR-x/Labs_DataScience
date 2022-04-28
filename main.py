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


def print_first_five_rows(dataset):
   print(dataset)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_path = 'data\Data2.csv'
    dataset = read_dataset(data_path)

    print_first_five_rows(dataset)

    print(dataset.info())

    dataset['GDP per capita'] = dataset['GDP per capita'].str.replace(',', '.').astype(float)
    dataset['CO2 emission'] = dataset['CO2 emission'].str.replace(',', '.').astype(float)
    dataset['Area'] = dataset['Area'].str.replace(',', '.').astype(float)

    print(dataset.info())

    print_first_five_rows(dataset)