import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from nltk.tokenize import RegexpTokenizer
import pymorphy2
import re
import csv
import io

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_colwidth', None)
desired_width = 1500
pd.set_option('display.width', desired_width)



def clean_data(dataset):
    for index in range(dataset.index[0], dataset.index[-1]):
        dataset.loc[index]['Title'] = re.sub(r"<[^>]+>", "", dataset.loc[index]['Title'])
        dataset.loc[index]['Body'] = re.sub(r"<[^>]+>", "", dataset.loc[index]['Body'])

        dataset.loc[index]['Title'] = re.sub('[^A-Za-z0-9А-Яа-яҐґЄєІіЇї\']+', " ", dataset.loc[index]['Title'])
        dataset.loc[index]['Body'] = re.sub('[^A-Za-z0-9А-Яа-яҐґЄєІіЇї\']+', " ", dataset.loc[index]['Body'])

        dataset.loc[index]['Title'] = dataset.loc[index]['Title'].lower()
        dataset.loc[index]['Body'] = dataset.loc[index]['Body'].lower()

    return dataset


def get_stop_words():
    stopwords = []
    filename = 'data/stop_words_ua.csv'
    with io.open(filename, 'r', encoding="utf-8") as file:
        for row in csv.reader(file):
            stopwords.append(row[0])
    return stopwords


def get_abbreviation(language):
    if language == 'ukrainian':
        return ['тис', 'грн', 'т.я', 'вул', 'cек', 'хв', 'обл', 'кв', 'пл', 'напр', 'гл', 'і.о', 'зам']


def tokenize_del_stop_worlds(dataset):
    nltk.download('punkt')
    stop_words = get_stop_words()

    for index in range(dataset.index[0], dataset.index[-1]):
        tokens = nltk.word_tokenize(dataset.loc[index]['Title'])
        text = [word for word in tokens if word not in stop_words]
        dataset['Title'].loc[index] = text

        tokens = nltk.word_tokenize(dataset.loc[index]['Body'])
        text = [word for word in tokens if word not in stop_words]
        dataset['Body'].loc[index] = text

    return dataset




if __name__ == '__main__':
    data_path = 'data/ukr_text.csv'
    dataset = pd.read_csv(data_path, sep=',', encoding='utf-8')

    print(dataset.info())
    print(dataset.head())


        #1. очистка 2. токенізація
    clean_data(dataset)
    print(dataset.head())
    tokenize_del_stop_worlds(dataset)
    print(dataset.head())
