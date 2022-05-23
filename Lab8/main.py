import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_colwidth', None)
desired_width = 1500
pd.set_option('display.width', desired_width)



if __name__ == '__main__':
    data_path = 'data/ukr_text.csv'
    dataset = pd.read_csv(data_path, sep=',', encoding='utf-8')

    print(dataset.info())
    print(dataset.head())
