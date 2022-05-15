import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
desired_width = 300
pd.set_option('display.width', desired_width)



if __name__ == '__main__':
    data_path = 'data\\titanic.csv'
    dataset = pd.read_csv(data_path, sep=',', encoding='cp1252', decimal='.')

    print(dataset.info())
    print(dataset.head(10))

    dataset['Pclass'] = dataset['Pclass'].astype(str)
    print(dataset.info())

    dataset = dataset.drop(columns=['PassengerId'])
    dataset = dataset.drop(columns=['Name'])

    print(dataset.info())
    
