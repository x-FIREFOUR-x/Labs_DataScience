import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

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

def is_Normal_ShpiroWilk(column, a = 0.05):
    if len(column) < 3:
        return True
    st, pvalue = stats.shapiro(column)
    return pvalue > a


def delete_outliers(dataset, column_label):
    Q1 = dataset[column_label].quantile(0.25)
    Q3 = dataset[column_label].quantile(0.75)
    QR = Q3 - Q1
    dataset = dataset.drop(dataset[(dataset[column_label] < (Q1 - 1.5 * QR)) |
                                   (dataset[column_label] > (Q3 + 1.5 * QR))]
                           .index)
    return dataset



if __name__ == '__main__':

    #main task
    data_path = 'data\winequality-red.csv'
    dataset = pd.read_csv(data_path, sep=',', encoding='cp1252')

    print(dataset.info())
    print(dataset.head(10))

    print_neg_elements_dataset(dataset)
    print_nan_elements_dataset(dataset)

    datasetcorr = dataset.corr()
    print(datasetcorr)

    for column in dataset.columns:
        print(is_Normal_ShpiroWilk(dataset[column]))

    for column in dataset.columns:
        dataset = delete_outliers(dataset, column)

    X = dataset.iloc[:, :11].to_numpy()
    Y = dataset['quality'].to_numpy()

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=0)

        #Побудуємо лінійну модель для прогнозу quility
    lineRegresion = LinearRegression().fit(Xtrain, Ytrain)

        #Побудуємо поліноміальну модель для прогнозу quility 2-4 порядків
    Regresions = [lineRegresion]
    for i in range(2, 4):
        Regresions.append(make_pipeline(PolynomialFeatures(degree=i), LinearRegression()))
        Regresions[i - 1].fit(Xtrain, Ytrain)

        #Обрахуємо Середня квадратичну помилку(mse) коефіцієнт детермінації(r2)
    mse = []
    r2 = []
    for i in Regresions:
        pred = i.predict(Xtest)
        mse.append(mean_squared_error(Ytest, pred))
        r2.append(r2_score(Ytest, pred))

    degree = list(range(1, 4))
    plt.plot(degree, mse, color='red')
    plt.plot(degree, r2, color='blue')
    plt.show()