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
def print_dataset(dataset):
   print(dataset)


    #конвертувати тип даних колонки column_name датафрейма з string в float
def convert_column_str_to_float(dataset, column_name):
    dataset[column_name] = dataset[column_name]\
        .str.replace(',', '.')\
        .astype(float)


    #замінити NaN на середні значення
def replace_nan_to_avarege(dataset):
    return dataset.fillna(dataset.mean())


    #замінити від'ємні значення в колонці column_name на їх абсолютні значення
def abs_date(dataset, column_name):
     dataset[column_name] = dataset[column_name].abs()


    #Виправити помилки в даних (типиб заміна пустих і відємних значень)
def correction_error_in_date(dataset):
    convert_column_str_to_float(dataset, 'GDP per capita')
    convert_column_str_to_float(dataset, 'CO2 emission')
    convert_column_str_to_float(dataset, 'Area')

    dataset = replace_nan_to_avarege(dataset)

    abs_date(dataset, 'GDP per capita')
    abs_date(dataset, 'Populatiion')
    abs_date(dataset, 'CO2 emission')
    abs_date(dataset, 'Area')

    return dataset


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

def hist(dataset):
    footer, hists = plt.subplots(1, 4, figsize=(16, 6))

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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_path = 'data\Data2.csv'
    dataset = read_dataset(data_path)

    dataset = correction_error_in_date(dataset)

    print_dataset(dataset)
    print(dataset.info())

    boxplot(dataset)

    hist(dataset)

