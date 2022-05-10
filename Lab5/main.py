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
        if dataset[col_name].dtype != 'object':
            print(f'{col_name} neg values:\n', dataset[dataset[col_name] < 0])


def print_nan_elements_dataset(dataset):
    for col_name in dataset.columns:
        if dataset[col_name].dtype != 'object':
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
    '''
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
    '''


    #additional task
    data_path2 = 'data\Data4.csv'
    dataset2 = pd.read_csv(data_path2, sep=';', encoding='windows-1251', decimal=',')
    dataset2.rename(columns={'Unnamed: 0': 'Country'})

    print(dataset2.info())
    print(dataset2.head(10))

    print_neg_elements_dataset(dataset2)
    print_nan_elements_dataset(dataset2)

    datasetcorr2 = dataset2.corr()
    print(datasetcorr2)

    pd.plotting.scatter_matrix(dataset2, figsize=(10, 10))
    plt.show()

        # побудуємо регресійні моделі
    Y = dataset2['Cql']

    LRegresions = []
    LRegresions.append(LinearRegression().fit(dataset2['Ie'].to_numpy().reshape(-1, 1), Y))
    LRegresions.append(LinearRegression().fit(dataset2['Iec'].to_numpy().reshape(-1, 1), Y))
    LRegresions.append(LinearRegression().fit(dataset2['Is'].to_numpy().reshape(-1, 1), Y))
    LRegresions.append(LinearRegression().fit(dataset2[['Ie', 'Iec']], Y))
    LRegresions.append(LinearRegression().fit(dataset2[['Ie', 'Is']], Y))
    LRegresions.append(LinearRegression().fit(dataset2[['Iec', 'Is']], Y))
    LRegresions.append(LinearRegression().fit(dataset2[['Ie', 'Iec', 'Is']], Y))

    PRegresions = []
    PRegresions.append(make_pipeline(PolynomialFeatures(degree=2), LinearRegression()))
    PRegresions[0].fit(dataset2['Ie'].to_numpy().reshape(-1, 1), Y)
    PRegresions.append(make_pipeline(PolynomialFeatures(degree=2), LinearRegression()))
    PRegresions[1].fit(dataset2['Iec'].to_numpy().reshape(-1, 1), Y)
    PRegresions.append(make_pipeline(PolynomialFeatures(degree=2), LinearRegression()))
    PRegresions[2].fit(dataset2['Is'].to_numpy().reshape(-1, 1), Y)
    PRegresions.append(make_pipeline(PolynomialFeatures(degree=2), LinearRegression()))
    PRegresions[3].fit(dataset2[['Ie', 'Iec']], Y)
    PRegresions.append(make_pipeline(PolynomialFeatures(degree=2), LinearRegression()))
    PRegresions[4].fit(dataset2[['Ie', 'Is']], Y)
    PRegresions.append(make_pipeline(PolynomialFeatures(degree=2), LinearRegression()))
    PRegresions[5].fit(dataset2[['Iec', 'Is']], Y)
    PRegresions.append(make_pipeline(PolynomialFeatures(degree=2), LinearRegression()))
    PRegresions[6].fit(dataset2[['Ie', 'Iec', 'Is']], Y)


        #протестуєм моделі
    dstest = pd.read_csv('data/Data4t.csv', encoding='windows-1251', sep=';', decimal=',')
    dstest.rename(columns={'Unnamed: 0': 'Country'})

    testPredicts = []

    testPredicts.append(LRegresions[0].predict(dstest['Ie'].to_numpy().reshape(-1, 1)))
    testPredicts.append(LRegresions[1].predict(dstest['Iec'].to_numpy().reshape(-1, 1)))
    testPredicts.append(LRegresions[2].predict(dstest['Is'].to_numpy().reshape(-1, 1)))
    testPredicts.append(LRegresions[3].predict(dstest[['Ie', 'Iec']]))
    testPredicts.append(LRegresions[4].predict(dstest[['Ie', 'Is']]))
    testPredicts.append(LRegresions[5].predict(dstest[['Iec', 'Is']]))
    testPredicts.append(LRegresions[6].predict(dstest[['Ie', 'Iec', 'Is']]))

    testPredicts.append(PRegresions[0].predict(dstest['Ie'].to_numpy().reshape(-1, 1)))
    testPredicts.append(PRegresions[1].predict(dstest['Iec'].to_numpy().reshape(-1, 1)))
    testPredicts.append(PRegresions[2].predict(dstest['Is'].to_numpy().reshape(-1, 1)))
    testPredicts.append(PRegresions[3].predict(dstest[['Ie', 'Iec']]))
    testPredicts.append(PRegresions[4].predict(dstest[['Ie', 'Is']]))
    testPredicts.append(PRegresions[5].predict(dstest[['Iec', 'Is']]))
    testPredicts.append(PRegresions[6].predict(dstest[['Ie', 'Iec', 'Is']]))

    test_predictions = np.array(testPredicts)

    print(np.sum((test_predictions - dstest['Cql'].to_numpy()) ** 2, axis=1).argmin())