import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
desired_width = 300
pd.set_option('display.width', desired_width)




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


    pl_covid_ds = covid_ds['Cases'][['Poland']]
    pl_covid_ds.describe()

    ukr_covid_ds = covid_ds['Cases'][['Ukraine']]
    ukr_covid_ds.describe()

    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    ax[0].set_title('Ukraine')
    ax[0].grid('-')
    ax[0].hist(ukr_covid_ds)

    ax[1].set_title('Poland')
    ax[1].grid('-')
    ax[1].hist(pl_covid_ds)

    plt.show()

