import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
from sklearn.model_selection import train_test_split
import seaborn as sbn

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
desired_width = 300
pd.set_option('display.width', desired_width)


    #теплова карта пропущених значень в датасеті
def check_nan(dataset):
    plt.figure(figsize=(14, 10))

    plt.title('Теплова карта пропущених значень')

    is_patch = patch.Patch(color='black', label='Присутнє')
    not_patch = patch.Patch(color='red', label='Відсутнє')
    plt.legend(handles=[is_patch, not_patch], bbox_to_anchor=(1, 1), loc='upper left')

    colours = ['black', 'red']
    sbn.heatmap(dataset.isna(), cbar=False, cmap=sbn.color_palette(colours))

    plt.show()


    #кількості і відсотк пропущених значень в колонках датасета
def stats_nan_data(data):
    count = data.isnull().sum().sort_values(ascending=False)
    percent = (count / (data.count() + data.isnull().sum()) * 100).sort_values(ascending=False)
    missing = pd.concat([count, percent], axis=1, keys=['Count NaN', 'Percent NaN'])
    return missing


if __name__ == '__main__':
    data_path = 'data\\titanic.csv'
    dataset = pd.read_csv(data_path, sep=',', encoding='cp1252', decimal='.')

    print(dataset.info())
    print(dataset.head(2))

    dataset['Pclass'] = dataset['Pclass'].astype(str)
    print(dataset.info())

    dataset = dataset.drop(columns=['PassengerId'])
    dataset = dataset.drop(columns=['Name'])
    dataset = dataset.drop(columns=['Ticket'])


    check_nan(dataset)
    missing_data = stats_nan_data(dataset)
    print(missing_data.head(10))
    dataset = dataset.drop(columns=['Cabin'])



