import matplotlib.image as mpimg
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import geopandas as geopd
import numpy as np
from scipy.spatial import distance
import math

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
desired_width = 320
pd.set_option('display.width', desired_width)


    #зчитати датасет в датафрейм
def read_dataset(path, dec='.'):
    data = pd.read_csv(path, sep=';', decimal=dec, encoding='cp1252')
    return data


    #конвертувати тип даних колонки column_name датафрейма з string в float
def convert_column_str_to_float(dataset, column_name):
    dataset[column_name] = dataset[column_name]\
        .str.replace(',', '.')\
        .astype(float)


    #замінити NaN на середні значення
def replace_nan_to_avarege(dataset):
    return dataset.fillna(dataset.mean())


    #вивести негативні елементи колонок
def print_neg_elements_dataset(dataset):
    print('GDP per capita negative values:\n', dataset[dataset['GDP per capita'] < 0])
    print('\nPopulation negative values:\n', dataset[dataset['Populatiion'] < 0])
    print('\nCO2 emission negative values:\n', dataset[dataset['CO2 emission'] < 0])
    print('\nArea negative values:\n', dataset[dataset['Area'] < 0])


    #замінити від'ємні значення в колонці column_name на їх абсолютні значення
def abs_date(dataset, column_name):
     dataset[column_name] = dataset[column_name].abs()


def hist(dataset):
    footer, hists = plt.subplots(1, 4, figsize=(14, 6))
    footer.suptitle('Гістограми: ', fontsize=20)

    hists[0].set_title('GDP per capita')
    hists[0].hist(dataset['GDP per capita'])

    hists[1].set_title('Population')
    hists[1].hist(dataset['Populatiion'])

    hists[2].set_title('CO2 emission')
    hists[2].hist(dataset['CO2 emission'])

    hists[3].set_title('Area')
    hists[3].hist(dataset['Area'])

    plt.show()


    #Перевірка нормальності розподілу за тестом Шапіро-Уілка
def test_normality_shapiro(data, alpha = 0.05):
    statistic, pvalue = stats.shapiro(data)
    print('Шапіро-Уілка', 'Statistic=%.4f, pvalue=%.4f' % (statistic, pvalue))
    if pvalue > alpha:
        print('Дані відповідають нормальному розподілу')
    else:
        print('Дані не відповідають нормальному розподілу')


    # Перевірка нормальності розподілу за тестом Д'Агостіно-Пірсона
def test_normality(data, alpha=0.05):
    statistic, pvalue = stats.normaltest(data)
    print('Агостіно-Пірсона', 'Statistic=%.4f, pvalue=%.4f' % (statistic, pvalue))
    if pvalue > alpha:
        print('Дані відповідають нормальному розподілу')
    else:
        print('Дані не відповідають нормальному розподілу')


    #Перевірка нормальності розподілу CO2 emission по регіонам
def test_normality_CO2_regions(dataset):
    regions = pd.unique(dataset['Region'])

    for region in regions:
        #print(region)
        #print(dataset[dataset['Region'] == region]['CO2 emission'])
        print('\nПеревірка для регіону:', region)
        emissions_in_region = dataset[dataset['Region'] == region]['CO2 emission']
        try:
            test_normality_shapiro(emissions_in_region)
        except ValueError as mes:
            print(str(mes))

        try:
            test_normality(emissions_in_region)
        except ValueError as mes:
            print(str(mes))

def check_median(dataset, column, alpha = 0.05):
    N = len(dataset[column])
    G = dataset[column].std()
    df = N - 1
    Mz = dataset[column].median()
    A = dataset[column].mean()

    t = abs(Mz - A) / (G / math.sqrt(N))
    test_t = stats.t.sf(t, df)
    if test_t > alpha / 2:
        print(column, ': Гіпотеза підтверджується')
    else:
        print(column, ': Гіпотеза відхиляється')


    #Кругова діаграма населення по регіонам
def circle_diagram_popul_in_regions(dataset):
    regions = pd.unique(dataset['Region'])

    title, diagrams = plt.subplots(figsize=(8, 6))
    diagrams.pie(dataset.groupby('Region').sum()['Populatiion'],
                 labels=regions, autopct='%.3f%%')
    title.suptitle('Населення по регіонам:', fontsize=20)

    plt.show()

    #Відобразити карту України з бульбашками що відповідають населенню
def create_map(img_map, cities_coordinates, cities_population):
    footer, map = plt.subplots(figsize=(8, 6))
    footer.suptitle('Україна: ', fontsize=20)

    map.imshow(img_map)
    map.scatter(
        cities_coordinates[:, 0],
        cities_coordinates[:, 1],
        s=cities_population * 1000,
        c='red',
        alpha=0.3,
        linewidth=2
    )
    map.axis('off')
    plt.show()


    #Визначення між якими 2 містами з заданого списку найбільша відстань на карті
def max_distance(cities_name, cities_coordinates, img_map):
    distances = distance.cdist(cities_coordinates, cities_coordinates, 'euclidean')

    city1, city2 = np.unravel_index(distances.argmax(), distances.shape)

    distance_pixel = distances[city1, city2]

    width_ukr_km = 1316
    distance_km = distances[city1, city2] * (width_ukr_km / img_map.shape[1])

    print(f'\nМіж містами {cities_name[city1]} і {cities_name[city2]} '
          f'найбільша відстань: {distance_pixel:.3f} пікселів або {distance_km:.3f} км')


    #Створити картограму по даним колонки(column) геодатасету(geodataset) з назвою(title)
def create_cartography(geodataset, column, title='Картограма'):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    geodataset.plot(column=column, ax=ax, legend=True,
                    legend_kwds={'orientation': "horizontal"},
                    missing_kwds={
                        "color": "lightgrey",
                        "edgecolor": "red",
                        "label": "Missing values",
                    },
                    cmap='gnuplot', edgecolor='white', linewidth=0.2)
    fig.suptitle(title)
    ax.axis('off')
    plt.show()


    #Розрахувати коефіцієнт кореляції
def correlation_coefficient(dataset_dpp, dataset_gdp, geom):
    dpp = dataset_dpp.iloc[:, -5:]
    gdp = dataset_gdp.iloc[:, -5:]

    correlations = pd.DataFrame()
    correlations['Correlation'] = dpp.corrwith(gdp, axis=1)
    correlations['Name'] = dataset_dpp['Name']

    correlations = pd.merge(geom, correlations, on=['Name'])

    create_cartography(correlations, 'Correlation', "Коефіцієнт кореляції")



if __name__ == '__main__':
    data_path = 'data\Data2.csv'
    dataset = read_dataset(data_path)

    convert_column_str_to_float(dataset, 'GDP per capita')
    convert_column_str_to_float(dataset, 'CO2 emission')
    convert_column_str_to_float(dataset, 'Area')

    dataset = replace_nan_to_avarege(dataset)

    print_neg_elements_dataset(dataset)

    abs_date(dataset, 'GDP per capita')
    abs_date(dataset, 'Area')

    print(dataset.head(10))
    print(dataset.info())

    dataset.describe()


        #2 перевірити чи параметри розподілені за нормальним законом
    dataset.hist(figsize=(12, 10))
    plt.show()

    print('\nперевірка для GDP per capita:')
    test_normality(dataset['GDP per capita'])
    test_normality_shapiro(dataset['GDP per capita'])

    print('\nперевірка для Populatiion:')
    test_normality(dataset['Populatiion'])
    test_normality_shapiro(dataset['Populatiion'])

    print('\nперевірка для CO2 emission:')
    test_normality(dataset['CO2 emission'])
    test_normality_shapiro(dataset['CO2 emission'])

    print('\nперевірка для Area:')
    test_normality(dataset['Area'])
    test_normality_shapiro(dataset['Area'])


        #3 Перевірка середніх та медіан на значимість
    print('\n')
    check_median(dataset, 'GDP per capita')
    check_median(dataset, 'Populatiion')
    check_median(dataset, 'CO2 emission')
    check_median(dataset, 'Area')


        #4 визначення в якому регіоні розподіл викидів СО2 найбільш близький до нормального
    dataset['CO2 emission'].hist(by=dataset['Region'], figsize=(15, 20))
    plt.show()

    test_normality_CO2_regions(dataset)


        #5 кругова діаграма населення по регіонах
    circle_diagram_popul_in_regions(dataset)


    #Додаткове 1

    IMG_PATH = 'data/Maps/Ukraine.jpg'
    img_map = mpimg.imread(IMG_PATH)

    cities_name = ['Київ', 'Львів', 'Луцьк', 'Чернівці', 'Тернопіль']
    cities_coordinates = np.array([(386, 145), (90, 188), (156, 118), (175, 294), (162, 202)])
    cities_population = np.array([2.884, 0.721, 0.213, 0.285, 0.216])

    create_map(img_map, cities_coordinates, cities_population)

    max_distance(cities_name, cities_coordinates, img_map)


    #Додаткове 3

    geom = geopd.read_file('data/UKR_ADM1.shp')

    dataset_ukr_dpp = pd.read_csv('data/ukr_DPP.csv', sep=';', decimal=',', encoding='windows-1251', header=1)
    dataset_ukr_gdp = pd.read_csv('data/ukr_GDP.csv', sep=';', decimal=',', encoding='windows-1251', header=1)

    geodataset_dpp = geopd.GeoDataFrame(pd.merge(geom, dataset_ukr_dpp))
    geodataset_gdp = geopd.GeoDataFrame(pd.merge(geom, dataset_ukr_gdp))

    geodataset_gdp['2012'] = geodataset_gdp['2012'].astype(float)
    geodataset_gdp['2013'] = geodataset_gdp['2013'].astype(float)

    create_cartography(geodataset_dpp, '2016', "Прибуток населення на одну особу (2016 рік)")
    create_cartography(geodataset_gdp, '2016', "Валовий регіональний продукт (2016 рік)")

    correlation_coefficient(dataset_ukr_dpp, dataset_ukr_gdp, geom)
