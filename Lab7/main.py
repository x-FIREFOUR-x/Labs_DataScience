import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.tsa.api as smt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
desired_width = 300
pd.set_option('display.width', desired_width)


    #згладжування за допомогою ковзаючого середнього
def plot_moving_average(series, n):
    rolling_mean = series.rolling(window=n).mean()
    plt.figure(figsize=(15, 5))
    plt.title(f'Moving average\n window size = {n}')
    plt.plot(rolling_mean, c='orange', label='Rolling mean trend')
    plt.plot(series[n:], label='Actual values')
    plt.legend(loc='upper left')
    #plt.grid(True)
    plt.show()


    #Перевірка ряду на стаціонарність за допомогою доповненого тесту Дікі-Фуллера:
def dickey_fuller_test(series):
    test = smt.adfuller(series, autolag='AIC')
    print('adf: ', test[0])
    print('p-value: ', test[1])
    print('Critical values: ', test[4])
    if test[0] > test[4]['5%']:
        print('Наявні одиничні корені, ряд не стаціонарний.')
    else:
        print('Одиничні корені відсутні, ряд є стаціонарним.')




if __name__ == '__main__':

    #Основне завдання 1
    data_path = 'data\\Covid.csv'
    covid_ds = pd.read_csv(data_path, sep=',', encoding='cp1252', decimal='.')

    print(covid_ds.info())
    print(covid_ds.head(2))

    covid_ds = covid_ds.drop(columns=
                           ['Continent', 'Latitude', 'Longitude', 'Average temperature per year',
                            'Hospital beds per 1000 people', 'Medical doctors per 1000 people',
                            'GDP/Capita', 'Population', 'Median age',
                            'Population aged 65 and over (%)', 'Daily tests', 'Deaths'
                            ])
    print(covid_ds.info())
    print(covid_ds.head(2))

    print((covid_ds.groupby(['Entity']))['Entity'].head(1))
    covid_ds = covid_ds[(covid_ds['Entity'] == 'Ukraine') | (covid_ds['Entity'] == 'Poland')]
    print((covid_ds.groupby(['Entity']))['Entity'].head(1))

    print('negative values:\n', covid_ds[covid_ds['Cases'] < 0])

    print(covid_ds.head(5))

    covid_ds = pd.pivot_table(covid_ds, values=['Cases'], index=['Date'], columns=['Entity'])
    print(covid_ds.info())
    print(covid_ds['Cases']['Ukraine'].head(10))


    fig, ax = plt.subplots(figsize=(15, 10))
    covid_ds['Cases'][['Ukraine', 'Poland']].plot(ax=ax)
    plt.title('Часова динаміка Covid-19 в Україні та Польщі')
    ax.grid()
    plt.show()


    ukr_covid_ds = covid_ds['Cases'][['Ukraine']]
    ukr_covid_ds.describe()

    pl_covid_ds = covid_ds['Cases'][['Poland']]
    pl_covid_ds.describe()

    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    ax[0].set_title('Ukraine')
    ax[0].grid('-')
    ax[0].hist(ukr_covid_ds)

    ax[1].set_title('Poland')
    ax[1].grid('-')
    ax[1].hist(pl_covid_ds)

    plt.show()


    plot_moving_average(ukr_covid_ds, 5)
    plot_moving_average(ukr_covid_ds, 10)
    plot_moving_average(ukr_covid_ds, 20)

    plot_moving_average(pl_covid_ds, 5)
    plot_moving_average(pl_covid_ds, 10)
    plot_moving_average(pl_covid_ds, 20)

    '''
    decomposition = smt.seasonal_decompose(ukr_covid_ds[~ukr_covid_ds.isna()])
    fig = decomposition.plot()
    fig.set_size_inches(15, 10)
    plt.show()
    '''

    fig, ax = plt.subplots(2, figsize=(15, 10))
    fig.suptitle('Ukraine: ', fontsize=20)
    ax[0] = plot_acf(ukr_covid_ds[~ukr_covid_ds.isna()], ax=ax[0], lags=120)
    ax[1] = plot_pacf(ukr_covid_ds[~ukr_covid_ds.isna()], ax=ax[1], lags=120)
    plt.show()

    fig, ax = plt.subplots(2, figsize=(15, 10))
    fig.suptitle('Poland', fontsize=20)
    ax[0] = plot_acf(pl_covid_ds[~pl_covid_ds.isna()], ax=ax[0], lags=120)
    ax[1] = plot_pacf(pl_covid_ds[~pl_covid_ds.isna()], ax=ax[1], lags=120)
    plt.show()

    dickey_fuller_test(ukr_covid_ds[~ukr_covid_ds.isna()])
    #dickey_fuller_test(pl_covid_ds[~pl_covid_ds.isna()])


