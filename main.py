import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
desired_width = 320
pd.set_option('display.width', desired_width)


    #зчитати дасет в датафрейм
def read_dataset(path):
    data = pd.read_csv(path, sep=';', encoding='cp1252')
    return data


    #вивести датасет
def print_first_n_dataset(dataset, n):
   print(dataset.head(n))


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


    #побудувати діаграми розмаху
def boxplot(dataset):
    footer,\
    diagrams = plt.subplots(1, 4, figsize=(12, 6))

    footer.suptitle('Діаграми розмаху: ', fontsize=20)

    diagrams[0].set_title('GDP per capita')
    diagrams[0].boxplot(dataset['GDP per capita'])

    diagrams[1].set_title('Population')
    diagrams[1].boxplot(dataset['Populatiion'])

    diagrams[2].set_title('CO2 emission')
    diagrams[2].boxplot(dataset['CO2 emission'])

    diagrams[3].set_title('Area')
    diagrams[3].boxplot(dataset['Area'])

    plt.show()


    #побудувати гістограми
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


    #країна з найбільшим ввп на людину (GDP per capita)
def country_max_gdp(dataset):
    id_row_max_gdp = dataset['GDP per capita'].idxmax()
    row_dataset_max_gdp = dataset.loc[id_row_max_gdp]
    name_country_max_gdp = row_dataset_max_gdp['Country Name']
    print('Country with a max GDP per capita:', name_country_max_gdp)
    print('Index row in dataset with a max GDP per capita:', id_row_max_gdp)


    #країна з найменшою площею (Area)
def country_min_area(dataset):
    id_row_min_area = dataset['Area'].idxmin()
    row_dataset_min_area = dataset.loc[id_row_min_area]
    name_country_min_area = row_dataset_min_area['Country Name']
    print('Country with a min Area:', name_country_min_area)
    print('Index row in dataset with a min Area:', id_row_min_area)


    #регіон з найбільшою середньою площею країн
def region_average_area(dataset):
    average_area_regions = dataset.groupby(['Region']).mean()['Area']
    print('Average areas countries by regions:\n', average_area_regions)
    print('Region with max average areas countries by regions: ', average_area_regions.idxmax())


    #країна з найбільшим населенням
def country_max_population(dataset):
    id_row_max_population = dataset['Populatiion'].idxmax()
    row_max_population = dataset.loc[id_row_max_population]
    print('Country with a max Population:', row_max_population['Country Name'])
    print('Index row in dataset with a max Population:', id_row_max_population)


    #країна з найбільшим населенням в Європі
def country_max_populatuin_in_europe(dataset):
    id_row_max_population_in_europe = dataset[dataset['Region'] == 'Europe & Central Asia']['Populatiion'].idxmax()
    row_max_population_in_europe = dataset.loc[id_row_max_population_in_europe]
    print('Country with a max Population in Europe:', row_max_population_in_europe['Country Name'])
    print('Index row in dataset with a max Population in Europe:', id_row_max_population_in_europe)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_path = 'data\Data2.csv'
    dataset = read_dataset(data_path)

    convert_column_str_to_float(dataset, 'GDP per capita')
    convert_column_str_to_float(dataset, 'CO2 emission')
    convert_column_str_to_float(dataset, 'Area')

    dataset = replace_nan_to_avarege(dataset)

    #print_neg_elements_dataset(dataset)

    abs_date(dataset, 'GDP per capita')
    abs_date(dataset, 'Area')

    boxplot(dataset)

    hist(dataset)

    dataset['Density population'] = dataset['Populatiion'] / dataset['Area']

    #print_first_n_dataset(dataset, 217)
    #print(dataset.info())

    country_max_gdp(dataset)

    country_min_area(dataset)

    region_average_area(dataset)

    country_max_population(dataset)

    country_max_populatuin_in_europe(dataset)



