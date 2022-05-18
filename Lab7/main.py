import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    plt.grid(True)
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
    data_path = 'data\\DataCovid.csv'
    covid_ds = pd.read_csv(data_path, index_col=['date'], parse_dates=['date'])

    print(covid_ds.info())
    print(covid_ds.head(10))


    covid_ds = covid_ds[['country', 'daily_new_cases']]
    print(covid_ds.info())
    print(covid_ds.head(2))


    covid_ds = covid_ds.rename(columns={'country': 'Country','daily_new_cases': 'New cases'})


    print((covid_ds.groupby(['Country']))['Country'].head(1))
    covid_ds = covid_ds[(covid_ds['Country'] == 'Ukraine') | (covid_ds['Country'] == 'Romania')]
    print((covid_ds.groupby(['Country']))['Country'].head(1))

    print('negative values:\n', covid_ds[covid_ds['New cases'] < 0])


    print(covid_ds.head(5))

    covid_ds = covid_ds.pivot_table(values=['New cases'], index=['date'], columns=['Country'])
    print(covid_ds.info())
    print(covid_ds['New cases'].head(10))


    fig, ax = plt.subplots(figsize=(15, 10))
    covid_ds['New cases'][['Ukraine', 'Romania']].plot(ax=ax)
    plt.title('Часова динаміка Covid-19 в Україні та Румунії')
    ax.grid()
    plt.show()


    ukr_covid_ds = covid_ds['New cases']['Ukraine']
    ukr_covid_ds.describe()
    print(ukr_covid_ds.describe())

    rom_covid_ds = covid_ds['New cases']['Romania']
    rom_covid_ds.describe()
    print(rom_covid_ds.describe())


    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    ax[0].set_title('Ukraine')
    ax[0].grid('-')
    ax[0].hist(ukr_covid_ds)
    ax[1].set_title('Romania')
    ax[1].grid('-')
    ax[1].hist(rom_covid_ds)
    plt.show()


    plot_moving_average(ukr_covid_ds, 5)
    plot_moving_average(ukr_covid_ds, 10)
    plot_moving_average(ukr_covid_ds, 20)

    plot_moving_average(rom_covid_ds, 5)
    plot_moving_average(rom_covid_ds, 10)
    plot_moving_average(rom_covid_ds, 20)


    decomposition = smt.seasonal_decompose(ukr_covid_ds[~ukr_covid_ds.isna()])
    fig = decomposition.plot()
    fig.set_size_inches(15, 10)
    plt.show()

    decomposition = smt.seasonal_decompose(rom_covid_ds[~rom_covid_ds.isna()])
    fig = decomposition.plot()
    fig.set_size_inches(15, 10)
    plt.show()


    fig, ax = plt.subplots(2, figsize=(15, 10))
    fig.suptitle('Ukraine: ', fontsize=20)
    ax[0] = plot_acf(ukr_covid_ds[~ukr_covid_ds.isna()], ax=ax[0], lags=120)
    ax[1] = plot_pacf(ukr_covid_ds[~ukr_covid_ds.isna()], ax=ax[1], lags=120)
    plt.show()

    fig, ax = plt.subplots(2, figsize=(15, 10))
    fig.suptitle('Romania', fontsize=20)
    ax[0] = plot_acf(rom_covid_ds[~rom_covid_ds.isna()], ax=ax[0], lags=120)
    ax[1] = plot_pacf(rom_covid_ds[~rom_covid_ds.isna()], ax=ax[1], lags=120)
    plt.show()

    dickey_fuller_test(ukr_covid_ds[~ukr_covid_ds.isna()])
    dickey_fuller_test(rom_covid_ds[~rom_covid_ds.isna()])




    # Основне завдання 2
    data_path = 'data\\USD_UAH.csv'
    currencies_ds = pd.read_csv(data_path, index_col=['Date'], parse_dates=['Date'])

    currencies_ds = currencies_ds.sort_index(ascending=True)

    print(currencies_ds.info())
    print(currencies_ds.head(10))


    idx = pd.date_range(currencies_ds.index[0], currencies_ds.index[-1])
    currencies_ds = currencies_ds.reindex(idx)
    print(currencies_ds.head(10))

    for index in currencies_ds.index:
        if pd.isna(currencies_ds.loc[index, 'Price']):
            currencies_ds.loc[index] = currencies_ds.shift().loc[index]

    print(currencies_ds.head(10))


    print('negative values:\n', currencies_ds[currencies_ds['Price'] < 0])
    print('negative values:\n', currencies_ds[currencies_ds['Open'] < 0])
    print('negative values:\n', currencies_ds[currencies_ds['High'] < 0])
    print('negative values:\n', currencies_ds[currencies_ds['Low'] < 0])


    fig, ax = plt.subplots(figsize=(15, 12))
    currencies_ds[['Price']].plot(ax=ax, subplots=True)
    ax.grid(True)
    plt.show()


    plot_moving_average(currencies_ds['Price'], 10)
    plot_moving_average(currencies_ds['Price'], 20)
    plot_moving_average(currencies_ds['Price'], 30)


    price_decomposition = smt.seasonal_decompose(currencies_ds['Price'])
    fig = price_decomposition.plot()
    fig.set_size_inches(15, 10)
    plt.show()


    fig, ax = plt.subplots(2, figsize=(15, 10))
    ax[0] = plot_acf(currencies_ds['Price'], ax=ax[0], lags=100)
    ax[1] = plot_pacf(currencies_ds['Price'], ax=ax[1], lags=100)
    plt.show()


    dickey_fuller_test(currencies_ds['Price'])




    #Додаткове завдання
    weather_ds = pd.read_csv('data/seattleWeather_1948-2017.csv', index_col=['DATE'], parse_dates=['DATE'])

    print(weather_ds.info())
    print(weather_ds.head(10))

    print('nan values:\n', weather_ds[weather_ds['PRCP'].isna()])
    print('nan values:\n', weather_ds[weather_ds['TMAX'].isna()])
    print('nan values:\n', weather_ds[weather_ds['TMIN'].isna()])
    print('nan values:\n', weather_ds[weather_ds['RAIN'].isna()])

    weather_ds['PRCP'].fillna(0, inplace=True)
    weather_ds['RAIN'].fillna(False, inplace=True)


    fig, ax = plt.subplots(figsize=(15, 12))
    weather_ds[['PRCP', 'TMAX', 'TMIN']].plot(ax=ax, subplots=True)
    ax.grid(True)
    plt.show()

    fig, ax = plt.subplots(figsize=(15, 12))
    weather_ds[['PRCP', 'TMAX', 'TMIN']].loc[weather_ds.index[-1500:]].plot(ax=ax, subplots=True)
    ax.grid(True)
    plt.show()

    fig, ax = plt.subplots(figsize=(15, 12))
    weather_ds[['PRCP', 'TMAX', 'TMIN']].loc[weather_ds.index[-1500:]].resample('W').mean().plot(ax=ax, subplots=True)
    ax.grid(True)
    plt.show()

    prcp_decomposition = smt.seasonal_decompose(weather_ds['PRCP'].loc[weather_ds.index[-1500:]].resample('W').mean())
    fig = prcp_decomposition.plot()
    fig.set_size_inches(15, 10)
    plt.show()
    prcp_decomposition = smt.seasonal_decompose(weather_ds['TMAX'].loc[weather_ds.index[-1500:]].resample('W').mean())
    fig = prcp_decomposition.plot()
    fig.set_size_inches(15, 10)
    plt.show()
    prcp_decomposition = smt.seasonal_decompose(weather_ds['TMIN'].loc[weather_ds.index[-1500:]].resample('W').mean())
    fig = prcp_decomposition.plot()
    fig.set_size_inches(15, 10)
    plt.show()

    fig, ax = plt.subplots(2, figsize=(15, 10))
    ax[0] = plot_acf(weather_ds['PRCP'].loc[weather_ds.index[-1500:]].resample('W').mean(), ax=ax[0], lags=100)
    ax[1] = plot_pacf(weather_ds['PRCP'].loc[weather_ds.index[-1500:]].resample('W').mean(), ax=ax[1], lags=100)
    plt.show()
    fig, ax = plt.subplots(2, figsize=(15, 10))
    ax[0] = plot_acf(weather_ds['TMAX'].loc[weather_ds.index[-1500:]].resample('W').mean(), ax=ax[0], lags=100)
    ax[1] = plot_pacf(weather_ds['TMAX'].loc[weather_ds.index[-1500:]].resample('W').mean(), ax=ax[1], lags=100)
    plt.show()
    fig, ax = plt.subplots(2, figsize=(15, 10))
    ax[0] = plot_acf(weather_ds['TMIN'].loc[weather_ds.index[-1500:]].resample('W').mean(), ax=ax[0], lags=100)
    ax[1] = plot_pacf(weather_ds['TMIN'].loc[weather_ds.index[-1500:]].resample('W').mean(), ax=ax[1], lags=100)
    plt.show()

        #Конвертація фарангейта в градуси
    print(weather_ds.head(10))
    weather_ds['TMAX'] = (5 / 9) * (weather_ds['TMAX'] - 32)
    weather_ds['TMIN'] = (5 / 9) * (weather_ds['TMIN'] - 32)
    weather_ds[['TMIN', 'TMAX']].describe()
    print(weather_ds.head(10))

        #будуємо матрицю кореляції
    corr = weather_ds[['PRCP', 'TMAX', 'TMIN']].corr()
    print(corr)

    dickey_fuller_test(weather_ds['PRCP'].loc['2007-01-01':])

        #будуємо модель ARIMA для передбачення опадів на настпний рік
    train_data = weather_ds['PRCP'].loc['2007-01-01':]
    train_data.describe()
    print(train_data.head(10))

    model = smt.ARIMA(train_data, order=(3, 0, 1)).fit()
    model.summary()

    fig, ax = plt.subplots(figsize=(15, 10))
    train_data.loc[train_data.index[-300:]].plot(ax=ax)
    ax.vlines(train_data.index[-1], 0, 1.5, linestyle='--', color='r', label='Start of forecast')
    ax = model.plot_predict(train_data.index[-1], '2018-12-31', dynamic=True, plot_insample=False, ax=ax)
