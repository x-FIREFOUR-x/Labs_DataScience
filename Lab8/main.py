import pandas as pd
import nltk
import numpy as np

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

        dataset.loc[index]['Title'] = re.sub('[^A-Za-zА-Яа-яҐґЄєІіЇї\']+', " ", dataset.loc[index]['Title'])
        dataset.loc[index]['Body'] = re.sub('[^A-Za-zА-Яа-яҐґЄєІіЇї\']+', " ", dataset.loc[index]['Body'])

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


    #обрахувати tf_idf для number слів в корпусі
def tf_idf(total_dict, dictionarys, dataset , number):
    tf_ = tf(total_dict, dictionarys, dataset, number)
    idf_ = idf(total_dict, dictionarys, number)

    tf_idf = []
    for index in range(0, len(tf_)):
        arr = []
        val_idf = idf_[index]

        for val_tf in tf_[index]:
            val_tf_idf = val_tf * val_idf
            arr.append(val_tf_idf)

        tf_idf.append(arr)

    tf_idf = pd.DataFrame(tf_idf).T
    for i in tf_idf.columns:
        tf_idf = tf_idf.rename(columns={i: list(total_dict.keys())[i]})

    return tf_idf


    #мішок слів
def bag_of_word(total_diction, dictionarys):
    bag_word = pd.DataFrame(0, index=np.arange(len(dictionarys)), columns=total_diction.keys())

    for i in range(0, len(total_diction)):
        arr = []
        word_key = list(total_dict.keys())[i]

        for j in range(0, len(dictionarys)):
            if word_key in dictionarys[j].keys():
                arr.append(dictionarys[j][word_key])
            else:
                arr.append(0)

        bag_word[word_key] = arr

    return bag_word


    #отримати емоційне забарвлення тексту
def get_sentiment_for_text(text, dictionary):
    sentiment = 0
    for word in text:
        if word in dictionary:
            sentiment = sentiment + dictionary[word]

    if sentiment >= 0:
        return 'positive'
    else:
        return 'negative'


    #отримати емоційні забарвлення всіх текстів датафрейму
def sentiment_analysis(df, dictionary):
    sentiments = []
    for i in df['Body']:
        sentiment = get_sentiment_for_text(i, dictionary)
        sentiments.append(sentiment)

    return pd.DataFrame({'Text': df['Body'], 'Sentiment': sentiments})


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


        # 4 TF-IDF
    dictionarys = dictionerys(dataset, 'Body')
    total_dict = total_dictionary(dataset, 'Body')

    tf_idf = tf_idf(total_dict, dictionarys, dataset, 10)
    print(tf_idf.head(50))

        #5 Bag of Word
    bag_word = bag_of_word(total_dict, dictionarys)
    print(bag_word.head(2))


        #add task(sentiment analysis)
    dictionary_path = 'data/tone-dict-uk.tsv'
    tone_dictionary_df = pd.read_csv(dictionary_path, sep ='\t', names=['Word', 'Tone'])
    print(tone_dictionary_df.head())

    df_sentiment = sentiment_analysis(dataset, tone_dictionary_df)
    print(df_sentiment.head())
