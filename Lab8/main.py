import operator

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from pymorphy2 import MorphAnalyzer
import math

import pymorphy2
import re
import csv
import io


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_colwidth', None)
desired_width = 1500
pd.set_option('display.width', desired_width)



    #очистити дані від тегів символів розмітки і перевести в нижній регістер
def clean_data(dataset):
    for index in range(dataset.index[0], dataset.index[-1]):
        dataset.loc[index]['Title'] = re.sub(r"<[^>]+>", "", dataset.loc[index]['Title'])
        dataset.loc[index]['Body'] = re.sub(r"<[^>]+>", "", dataset.loc[index]['Body'])

        dataset.loc[index]['Title'] = re.sub('[^A-Za-z0-9А-Яа-яҐґЄєІіЇї\']+', " ", dataset.loc[index]['Title'])
        dataset.loc[index]['Body'] = re.sub('[^A-Za-z0-9А-Яа-яҐґЄєІіЇї\']+', " ", dataset.loc[index]['Body'])

        dataset.loc[index]['Title'] = dataset.loc[index]['Title'].lower()
        dataset.loc[index]['Body'] = dataset.loc[index]['Body'].lower()

    return dataset


    #отримати масив стоп слів
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


    #токенізація розбиття тестів на токени
def tokenize_del_stop_worlds(dataset):
    #nltk.download('punkt')
    stop_words = get_stop_words()

    for index in range(dataset.index[0], dataset.index[-1]):
        tokens = nltk.word_tokenize(dataset.loc[index]['Title'])
        text = [word for word in tokens if word not in stop_words]
        dataset['Title'].loc[index] = text

        tokens = nltk.word_tokenize(dataset.loc[index]['Body'])
        text = [word for word in tokens if word not in stop_words]
        dataset['Body'].loc[index] = text

    return dataset


    #лематизація (зведення слів до нормальної форми)
def lematization(dataset, morph):
    for index in range(dataset.index[0], dataset.index[-1]):

        text = dataset.loc[index]['Title']
        lem_words = []
        for word in text:
            lem_word = morph.parse(word)[0].normal_form
            lem_words.append(lem_word)
        dataset.loc[index]['Title'] = lem_words

        text = dataset.loc[index]['Body']
        lem_words = []
        for word in text:
            lem_word = morph.parse(word)[-1].normal_form
            lem_words.append(lem_word)
        dataset.loc[index]['Body'] = lem_words

    return dataset


def tfidf(text):
    tfidf = TfidfVectorizer().fit_transform(text)
    a = tfidf.toarray()
    return a


    #створити масив словників(один словник для одного корпусу)відповідаючих текстам в датафреймі
def dictionerys(dataset, column):
    arr_dictionary = []
    for index in range(dataset.index[0], dataset.index[-1]):
        diction = dictionary(dataset.loc[index][column])
        arr_dictionary.append(diction)
    return arr_dictionary


    #створити словник для тексту(одного корпусу)
def dictionary(text):
    dictionary = {}
    for word in text:
       if word in dictionary.keys():
           dictionary[word] += 1
       else:
           dictionary[word] = 1

    return dict(sorted(dictionary.items(), reverse=True, key=lambda x: x[1]))


    #створити загальний словник (кількість в всіх корпусах)
def total_dictionary(dataset, column):
    total_dict = {}
    for index in range(dataset.index[0], dataset.index[-1]):
        for word in dataset.loc[index][column]:
            if word in total_dict.keys():
                total_dict[word] += 1
            else:
                total_dict[word] = 1

    return dict(sorted(total_dict.items(), reverse=True, key=lambda x: x[1]))
    #return total_dict


    #обрахувати tf для number слів в корпусі, для кожного тексту
def tf(total_dict, dictionarys, dataset , number):
    _tf = []
    for i in range(0, number):
        arr_tf = []
        word_key = list(total_dict.keys())[i]

        for index in range(dataset.index[0], dataset.index[-1]):
            if word_key in dictionarys[index].keys():
                number_word = dictionarys[index][word_key]
                number_all_words = len(dataset['Body'].loc[index])
                tf = number_word / number_all_words
            else:
                tf = 0
            arr_tf.append(tf)

        _tf.append(arr_tf)

    return _tf


    #обрахувати idf для number слів в корпусі
def idf(total_dict, dictionarys, number):
    arr_idf = []
    number_sentenses = len(dictionarys)

    for i in range(0, number):
        number_sentenses_contain_world = 0
        word_key = list(total_dict.keys())[i]

        for index in range(0, number_sentenses):
            if word_key in dictionarys[index].keys():
                number_sentenses_contain_world += 1

        idf = number_sentenses / number_sentenses_contain_world
        arr_idf.append(idf)

    return arr_idf




if __name__ == '__main__':
    data_path = 'data/ukr_text.csv'
    dataset = pd.read_csv(data_path, sep=',', encoding='utf-8')

    print(dataset.info())
    print(dataset.head())

        #1 очистка, 2 токенізація
    clean_data(dataset)
    print(dataset.head())
    tokenize_del_stop_worlds(dataset)
    print(dataset.head())

        # 3 лематизація
    morph = pymorphy2.MorphAnalyzer(lang='uk')
    lematization(dataset, morph)
    print(dataset.head())

    #tfidf = TfidfVectorizer().fit_transform(text)
    #print(tfidf)
    #print(tfidf.toarray())



    dictionarys = dictionerys(dataset, 'Body')
    total_dict = total_dictionary(dataset, 'Body')

    print(dictionarys[0])
    print()
    print(total_dict)

    tf = tf(total_dict, dictionarys, dataset, 10)
    print(tf)
    print(len(tf))

    idf = idf(total_dict, dictionarys, 10)
    print(idf)
    print(len(idf))