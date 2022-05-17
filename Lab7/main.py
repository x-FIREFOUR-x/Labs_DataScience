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
    dataset = pd.read_csv(data_path, sep=',', encoding='cp1252', decimal='.')

    print(dataset.info())
    print(dataset.head(2))

    dataset = dataset.drop(columns=
                           ['Continent', 'Latitude', 'Longitude', 'Average temperature per year',
                            'Hospital beds per 1000 people', 'Medical doctors per 1000 people',
                            'GDP/Capita', 'Population', 'Median age',
                            'Population aged 65 and over (%)', 'Daily tests', 'Deaths'
                            ])
    print(dataset.info())
    print(dataset.head(2))

    print((dataset.groupby(['Entity']))['Entity'].head(1))
    dataset = dataset[(dataset['Entity'] == 'Ukraine') | (dataset['Entity'] == 'Poland')]
    print((dataset.groupby(['Entity']))['Entity'].head(1))

    print('negative values:\n', dataset[dataset['Cases'] < 0])

    print(dataset.head(5))

    dataset = pd.pivot_table(dataset, values=['Cases'], index=['Date'], columns=['Entity'])
    print(dataset.info())
    print(dataset['Cases']['Ukraine'].head(10))


    fig, ax = plt.subplots(figsize=(15, 10))
    dataset['Cases'][['Ukraine', 'Poland']].plot(ax=ax)
    plt.title('Часова динаміка Covid-19 в Україні та Польщі')
    ax.grid()
    plt.show()

