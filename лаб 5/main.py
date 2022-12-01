import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer

from sklearn import tree

from sklearn import svm

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def graph(x, y, classes, color, plot_support):

    colors_list = [color[str(i)] for i in classes]
    plt.scatter(x, y, c = colors_list, cmap=plt.cm.Paired)
    if plot_support:
        w = svm_model.coef_[0]
        a = -w[0] / w[1]
        xx = np.linspace(x.max(), x.min())
        yy = a * xx - (svm_model.intercept_[0]) / w[1]
        margin = 1 / np.sqrt(np.sum(svm_model.coef_**2))
        yy_down = yy - np.sqrt(1 + a**2) * margin
        yy_up = yy + np.sqrt(1 + a**2) * margin
        plt.plot(xx, yy, "k-")
        plt.plot(xx, yy_down, "k--")
        plt.plot(xx, yy_up, "k--")
    plt.show()

#dataset for Decission Tree
dataset = pd.read_csv("diabetes.csv")
#Подготовка данных к обучению
target = dataset["Outcome"]
dataset.drop("Outcome", axis = 1, inplace = True)
#dataset for SVM
dataset_glucose_bmi = dataset[["Glucose", "BMI"]]
#dataset for LDA
dataset_age_family = dataset[["DiabetesPedigreeFunction", "Age"]]

#Определение наиболее влияющих признаков
#categorial_features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insuline", "BMI", "DiabetesPedigreeFunction", "Age"]
#clf = RandomForestClassifier()
#clf.fit(dataset,target)
#plt.figure(figsize=(12,12))
#plt.bar(categorial_features, clf.feature_importances_)
#plt.xticks(rotation=45)
#plt.show()

#Разделение данных на выборки
#for Decission Tree
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(dataset, target, random_state=0)
#for SVM
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(dataset_glucose_bmi, target, random_state=0)
#for LDA
X_train3, X_test3, Y_train3, Y_test3 = train_test_split(dataset_age_family, target, random_state=0)

#Обучение на сырых данных
#for Decission Tree
tree_model = tree.DecisionTreeClassifier()
tree_model.fit (X_train1, Y_train1)
predictions = tree_model.predict(X_test1)
print("Accuracy: {}".format((tree_model.score(X_test1,Y_test1))*100))
#for SVM
svm_model = svm.SVC(kernel='linear')
svm_model.fit(X_train2, Y_train2)
predictions = svm_model.predict(X_test2)
print("Accuracy: {}".format((svm_model.score(X_test2,Y_test2))*100))
#for LDA
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train3, Y_train3)
predictions = lda_model.predict(X_test3)
print("Accuracy: {}".format((lda_model.score(X_test3,Y_test3))*100))

#Обработка данных
#for Decission Tree
scaler = StandardScaler().fit(X_train1)
scaler.transform(X_train1)
scaler.transform(X_test1)
scaler = Normalizer().fit(X_train1)
scaler.transform(X_train1)
scaler.transform(X_test1)
#for SVM
scaler = StandardScaler().fit(X_train2)
scaler.transform(X_train2)
scaler.transform(X_test2)
scaler = Normalizer().fit(X_train2)
scaler.transform(X_train2)
scaler.transform(X_test2)
#for LDA
scaler = StandardScaler().fit(X_train3)
scaler.transform(X_train3)
scaler.transform(X_test3)
scaler = Normalizer().fit(X_train3)
scaler.transform(X_train3)
scaler.transform(X_test3)

#Обучение на очищенных данных
#for Decission Tree
tree_model = tree.DecisionTreeClassifier()
tree_model.fit (X_train1, Y_train1)
predictions1 = tree_model.predict(X_test1)
print("Accuracy: {}".format((tree_model.score(X_test1,Y_test1))*100))
#for SVM
svm_model = svm.SVC(kernel='linear')
svm_model.fit(X_train2, Y_train2)
predictions2 = svm_model.predict(X_test2)
print("Accuracy: {}".format((svm_model.score(X_test2,Y_test2))*100))
#for LDA
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train3, Y_train3)
predictions3= lda_model.predict(X_test3)
print("Accuracy: {}".format((lda_model.score(X_test3,Y_test3))*100))

#Визуализация
#for Decission Tree
#fig = plt.figure(figsize=(100,100))
#_ = tree.plot_tree(tree_model, feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
#       'BMI', 'DiabetesPedigreeFunction', 'Age'], class_names = ['0', '1'], filled=True)
#fig.savefig("tree.png")

#for SVM
#colors = {'0': 'red', '1': 'blue'}
#graph(X_test2["Glucose"], X_test2["BMI"], Y_test2, colors, False)
#graph(X_test2["Glucose"], X_test2["BMI"], predictions2, colors, True)

#for LDA
#colors = {'0': 'red', '1': 'blue'}
#graph(X_test3["Age"], X_test3["DiabetesPedigreeFunction"], Y_test3, colors, False)
#graph(X_test3["Age"], X_test3["DiabetesPedigreeFunction"], predictions3, colors, False)
