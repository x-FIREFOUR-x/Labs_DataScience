import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import seaborn as sbn

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score

from sklearn.cluster import KMeans
import plotly.express as px
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
desired_width = 300
pd.set_option('display.width', desired_width)


    #теплова карта пропущених значень в датасеті
def check_nan(dataset):
    plt.figure(figsize=(14, 10))

    plt.title('Теплова карта пропущених значень')

    is_patch = patch.Patch(color='black', label='Присутнє')
    not_patch = patch.Patch(color='red', label='Відсутнє')
    plt.legend(handles=[is_patch, not_patch], bbox_to_anchor=(1, 1), loc='upper left')

    colours = ['black', 'red']
    sbn.heatmap(dataset.isna(), cbar=False, cmap=sbn.color_palette(colours))

    plt.show()


    #кількості і відсотк пропущених значень в колонках датасета
def stats_nan_data(data):
    count = data.isnull().sum().sort_values(ascending=False)
    percent = (count / (data.count() + data.isnull().sum()) * 100).sort_values(ascending=False)
    missing = pd.concat([count, percent], axis=1, keys=['Count NaN', 'Percent NaN'])
    return missing


if __name__ == '__main__':

        #Основне завдання
    data_path = 'data\\titanic.csv'
    dataset = pd.read_csv(data_path, sep=',', encoding='cp1252', decimal='.')

    print(dataset.info())
    print(dataset.head(2))

    dataset['Pclass'] = dataset['Pclass'].astype(str)
    print(dataset.info())

    dataset = dataset.drop(columns=['PassengerId'])
    dataset = dataset.drop(columns=['Name'])
    dataset = dataset.drop(columns=['Ticket'])


    check_nan(dataset)
    missing_data = stats_nan_data(dataset)
    print(missing_data.head(10))
    dataset = dataset.drop(columns=['Cabin'])

    dataset_train, dataset_test = train_test_split(dataset, test_size=0.2, random_state=1)

    dataset_train = dataset_train.fillna(dataset_train.mean())
    dataset_test = dataset_test.fillna(dataset_test.mean())

    dataset_train['Embarked'] = dataset_train['Embarked'].fillna(dataset_train['Embarked'].mode()[0])
    dataset_test['Embarked'] = dataset_test['Embarked'].fillna(dataset_train['Embarked'].mode()[0])

    dataset_train['Age'] = dataset_train['Age'].fillna(dataset_train['Age'].mode()[0])
    dataset_test['Age'] = dataset_test['Age'].fillna(dataset_train['Age'].mode()[0])

    missing_data = stats_nan_data(dataset_train)
    print(missing_data.head(10))
    missing_data = stats_nan_data(dataset_test)
    print(missing_data.head(10))


    features = pd.concat([dataset_train, dataset_test]).reset_index(drop=True)
    features = pd.get_dummies(features)
    dataset_train = features.iloc[:dataset_train.shape[0], :]
    dataset_test = features.iloc[dataset_train.shape[0]:, :]


    Xtrain = dataset_train.drop(columns='Survived')
    Ytrain = dataset_train['Survived']

    Xtest = dataset_test.drop(columns='Survived')
    Ytest = dataset_test['Survived']

    decision_tree = DecisionTreeClassifier(max_depth=3, random_state=1)
    tree_scores = cross_val_score(decision_tree, Xtrain, Ytrain, cv=5)
    print('Decision Tree cvs:', tree_scores.mean())
    decision_tree.fit(Xtrain, Ytrain)
    print('Decision Tree test:', decision_tree.score(Xtest, Ytest))

    random_forest = RandomForestClassifier(max_depth=5)
    forest_scores = cross_val_score(random_forest, Xtrain, Ytrain, cv=5)
    print('Random forest cvs:', forest_scores.mean())
    random_forest.fit(Xtrain, Ytrain)
    print('Random forest test:', random_forest.score(Xtest, Ytest))

    boost = AdaBoostClassifier(learning_rate=0.3)
    boost_scores = cross_val_score(boost, Xtrain, Ytrain, cv=5)
    print('AdaBoost cvs:', boost_scores.mean())
    boost.fit(Xtrain, Ytrain)
    AdaBoostClassifier(learning_rate=0.3)
    print('AdaBoost test:', boost.score(Xtest, Ytest))

    gradboost = GradientBoostingClassifier(learning_rate=0.1)
    gradboost_scores = cross_val_score(gradboost, Xtrain, Ytrain, cv=5)
    print('GradientBoosting cvs:', gradboost_scores.mean())
    gradboost.fit(Xtrain, Ytrain)
    GradientBoostingClassifier(learning_rate=0.1)
    print('GradientBoosting test:', gradboost.score(Xtest, Ytest))

    bagging = GradientBoostingClassifier(learning_rate=0.1)
    bagging_scores = cross_val_score(gradboost, Xtrain, Ytrain, cv=5)
    print('Bagging cvs:', bagging_scores.mean())
    bagging.fit(Xtrain, Ytrain)
    print('Bagging test:', bagging.score(Xtest, Ytest))



    #Додаткове завдання
    dataset2 = pd.read_csv('data/Data2.csv', sep=';', decimal=',', encoding='windows-1251')

    dataset2 = dataset2.rename(columns={'Populatiion': 'Population'})
    dataset2['GDP per capita'] = abs(dataset2['GDP per capita'])
    dataset2['Area'] = abs(dataset2['Area'])
    dataset2 = dataset2.fillna(dataset2.mean())

    dataset2['Population density'] = dataset2['Population'] / dataset2['Area']

    print(dataset2.info())

    features = dataset2[['GDP per capita', 'Population density']]


    kmeans_kwargs = {
        'init': 'random',
        'n_init': 10,
        'max_iter': 300,
        'random_state': 42
    }
    sse = []
    max_cluster = 10
    for k in range(1, max_cluster + 1):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(features)
        sse.append(kmeans.inertia_)

    plt.figure(figsize=(8, 8))
    plt.plot(range(1, max_cluster + 1), sse)
    plt.xticks(range(1, max_cluster + 1))
    plt.xlabel('Number of Clusters')
    plt.ylabel('SSE')
    plt.grid(linestyle='-')
    plt.show()

    knee = KneeLocator(range(1, max_cluster + 1), sse, curve='convex', direction='decreasing')
    print(f'Точка локтя: {knee.elbow}')


    kmeans = KMeans(
        init='random',
        n_clusters=4,
        n_init=10,
        max_iter=300,
        random_state=42
    )
    kmeans.fit(features)

        #візуалізуємо кластери:
    fig = px.scatter(
        dataset2, x='GDP per capita', y='Population density', color=kmeans.labels_,
        hover_data=['Country Name', 'Region'],
        width=800, height=600
    )
    fig.update(layout_coloraxis_showscale=False)
    fig.show()

        #гістограми через цикл
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    labels = dataset2.columns[2:]
    for i in range(len(labels)):
        ax_i = (i // 3, i % 3)
        axes[ax_i].set_title(labels[i])
        axes[ax_i].grid('-')
        axes[ax_i].hist(dataset2[labels[i]])
    fig.delaxes(axes[1][2])
    plt.show()