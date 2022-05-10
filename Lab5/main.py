import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
desired_width = 250
pd.set_option('display.width', desired_width)

def print_neg_elements_dataset(dataset):
    for col_name in dataset.columns:
        print(f'{col_name} neg values:\n', dataset[dataset[col_name] < 0])


def print_nan_elements_dataset(dataset):
    for col_name in dataset.columns:
        print(f'{col_name} nan values:\n', dataset[dataset[col_name] < 0])


def delete_emissions(dataset, column_label):
    Q1 = dataset[column_label].quantile(0.25)
    Q3 = dataset[column_label].quantile(0.75)
    IQR = Q3 - Q1
    dataset = dataset.drop(dataset[(dataset[column_label] < (Q1 - 1.5 * IQR)) |
                                   (dataset[column_label] > (Q3 + 1.5 * IQR))]
                           .index)
    return dataset


if __name__ == '__main__':

    data_path = 'data\winequality-red.csv'
    dataset = pd.read_csv(data_path, sep=',', encoding='cp1252')

    print(dataset.info())
    print(dataset.head(10))

    print_neg_elements_dataset(dataset)
    print_nan_elements_dataset(dataset)

    datasetcorr = dataset.corr()
    print(datasetcorr)

    for column in dataset:
        dataset = delete_emissions(dataset, column)
