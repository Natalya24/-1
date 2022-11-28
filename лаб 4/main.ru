import csv
import math
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#загрузка сsv-файла
def load_csv(filename):
    dataset = list()
    with open("/Users/natalya/Desktop/МИИ-4/"+filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        for row in lines:
            if not row:
                continue
            dataset.append(row)
    return dataset

#основная задача метрического классификатора - определить класс объекта тестовой выборки
    #определить значение ошибки (точности)

#определение меры сходства
#подсчет евклидова расстояния между двумя двумерными точками
def dist(instance1, instance2):
    return math.sqrt((int(instance1[1]) - int(instance2[1]))**2 + (int(instance1[2]) - int(instance2[2]))**2)

#определение k наиболее похожих соседей из обучающего набора для тестовой выборки
def getNeighbors(trainingData, testingData, k):
    neighborsData = []
    #для каждого значения тестируемой выборки определим меру сходства и занесем в список
    distData = []
    for i in range(len(trainingData)):
        distance = dist(testingData, trainingData[i])
        distData.append((trainingData[i], distance))
    #сортировка полученного списка расстояний по возрастанию
    def keyFuncSort(item):
        return item[1]
    distData.sort(key = keyFuncSort)
    #получение k наиболее подходящих соседей по расстоянию
    for x in range(k):
        neighborsData.append(distData[x][0])
    return neighborsData

#создание словаря классов с результатами "голосования" соседей - возвращает наиболее часто встречающееся значение выбранной метки
def getLabel(neighborsData):
    labels = {}
    for x in range(len(neighborsData)):
        labels[neighborsData[x][-1]] = labels.get(neighborsData[x][-1], 0) + 1
    def keyFuncSort(item):
        return item[1]
    labels = sorted(labels.items(), key = keyFuncSort, reverse = True)
    return labels[0][0]

def getAccuracy(testingData, predictions):
    correct = 0
    for x in range(len(testingData)):
        if int(testingData[x][-1]) == int(predictions[x]):
            correct += 1
    return (correct/float(len(testingData)))*100

def knn_sklearn(trainData, testData, classes_train, classes_testing, k):

    X_train = trainData
    X_testing = testData
    y_train = classes_train
    y_testing = classes_testing

    scalerror = StandardScaler()
    scalerror.fit(X_train)

    X_train = scalerror.transform(X_train)
    X_testing = scalerror.transform(X_testing)

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    # Предсказывание
    predictions = model.predict(X_testing)

    print('Параметры обучающей выборки')
    print(X_train)
    print('Параметры тестовой выборки')
    print(X_testing)
    print('Классы обучающей выборки')
    print(y_train)
    print('Классы тестовой выборки')
    print(y_testing)
    print('Предсказания')
    print(predictions)

    return X_train, X_testing, y_train, y_testing, predictions

def graph(sweetness, crunch, classes, color):

    color_list = [color[str(i)] for i in classes]

    plt.scatter(sweetness, crunch, c = color_list)
    plt.show()

#загрузка обучающей выборки
trainData = load_csv("data.csv")
#загрузка тестовой выборки
testData = load_csv("data2.csv")

k = 1

classes = []
for i in range(len(testData)):
    classes.append(testData[i][-1])

sweetness = []
for i in range(len(testData)):
    sweetness.append(int(testData[i][1]))

crunch = []
for i in range(len(testData)):
    crunch.append(int(testData[i][2]))

#Метрический классификатор
print("Значение k = ", k)
#предсказывание класса
predictions = []
for x in range(len(testData)):
    print("Значение выборки: ", testData[x])
    neighbors = getNeighbors(trainData, testData[x], k)
    print("Наилучшие соседи: ", neighbors)
    result = getLabel(neighbors)
    predictions.append(result)
    print("предсказываемый класс " + str(result) + " актуальный класс " + repr(testData[x][-1]))
accuracy = getAccuracy(testData, predictions)
print("Точность " + str(accuracy) + "%")

colors = {'0': 'red', '1': 'blue', '2': 'green'}

graph(sweetness, crunch, classes, colors)
graph(sweetness, crunch, predictions, colors)


classes_train = []
for i in range(len(trainData)):
    classes_train.append(trainData[i][-1])

classes_testing = []
for i in range(len(testData)):
    classes_testing.append(testData[i][-1])

sweetness_train = []
for i in range(len(trainData)):
    sweetness_train.append(int(trainData[i][1]))

crunch_train= []
for i in range(len(trainData)):
    crunch_train.append(int(trainData[i][2]))

trainData=np.array(list(zip(sweetness_train, crunch_train)), dtype=np.float64)

sweetness_test = []
for i in range(len(testData)):
    sweetness_test.append(int(testData[i][1]))

crunch_test= []
for i in range(len(testData)):
    crunch_test.append(int(testData[i][2]))

testData=np.array(list(zip(sweetness_test, crunch_test)), dtype=np.float64)

X_train, X_testing, y_train, y_testing, predictions = knn_sklearn(trainData, testData, classes_train, classes_testing, k)
print("Точность - " + str((accuracy_score(y_testing, predictions))*100.0) + " %")

graph(sweetness_test, crunch_test, y_testing, colors)
graph(sweetness_test, crunch_test, predictions, colors)

#повторение эксперимента с новым классом "Снеки" - '3'
#загрузка обучающей выборки
trainData = load_csv("datanew.csv")
#загрузка тестовой выборки
testData = load_csv("data1new.csv")

classes = []
for i in range(len(testData)):
    classes.append(testData[i][-1])

sweetness = []
for i in range(len(testData)):
    sweetness.append(int(testData[i][1]))

crunch = []
for i in range(len(testData)):
    crunch.append(int(testData[i][2]))

#Метрический классификатор
print("Значение k = ", k)
#предсказывание класса
predictions = []
for x in range(len(testData)):
    print("Значение выборки: ", testData[x])
    neighbors = getNeighbors(trainData, testData[x], k)
    print("Наилучшие соседи: ", neighbors)
    result = getLabel(neighbors)
    predictions.append(result)
    print("предсказываемый класс " + str(result) + " актуальный класс " + repr(testData[x][-1]))
accuracy = getAccuracy(testData, predictions)
print("Точность " + str(accuracy) + "%")

colors = {'0': 'red', '1': 'blue', '2': 'green', '3': 'yellow'}

graph(sweetness, crunch, classes, colors)
graph(sweetness, crunch, predictions, colors)


classes_train = []
for i in range(len(trainData)):
    classes_train.append(trainData[i][-1])

classes_testing = []
for i in range(len(testData)):
    classes_testing.append(testData[i][-1])

sweetness_train = []
for i in range(len(trainData)):
    sweetness_train.append(int(trainData[i][1]))

crunch_train= []
for i in range(len(trainData)):
    crunch_train.append(int(trainData[i][2]))

trainData=np.array(list(zip(sweetness_train, crunch_train)), dtype=np.float64)

sweetness_test = []
for i in range(len(testData)):
    sweetness_test.append(int(testData[i][1]))

crunch_test= []
for i in range(len(testData)):
    crunch_test.append(int(testData[i][2]))

testData=np.array(list(zip(sweetness_test, crunch_test)), dtype=np.float64)

X_train, X_testing, y_train, y_testing, predictions = knn_sklearn(trainData, testData, classes_train, classes_testing, k)
print("Точность - " + str((accuracy_score(y_testing, predictions))*100.0) + " %")

graph(sweetness_test, crunch_test, y_testing, colors)
graph(sweetness_test, crunch_test, predictions, colors)
