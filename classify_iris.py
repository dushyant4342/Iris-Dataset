import pandas as pd
import numpy
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris

data=pd.read_csv('C:/Users/Dushyant Singh/Desktop/Iris/iris.csv')

data
data.shape
data.head()
data.tail()
len(data)
import matplotlib.pyplot as plt
data.columns
data.count()
data['SepalLengthCm'].min(),data['SepalLengthCm'].max()
data['SepalWidthCm'].min(),data['SepalWidthCm'].max()
data['PetalLengthCm'].min(),data['PetalLengthCm'].max()
data['PetalWidthCm'].min(),data['PetalWidthCm'].max()
data['SepalLengthCm'].value_counts().sort_index().plot(kind='bar')
data['SepalWidthCm'].value_counts().sort_index().plot(kind='bar')
data['PetalLengthCm'].value_counts().sort_index().plot(kind='bar')
data['PetalWidthCm'].value_counts().sort_index().plot(kind='bar')



data.describe()



X=data[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
Y=data["Species"]

X = X.to_numpy()
X.ndim

Y = Y.to_numpy()
Y.ndim

Y


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

model1 = RandomForestClassifier()
model1.fit(X_train, Y_train)
pred1 = model1.predict(X_test)
pred1
sklearn.metrics.accuracy_score(pred1,Y_test)*100
classification_report(Y_test, pred1)












model2=KNeighborsClassifier()
model2.fit(X_train,Y_train)
pred2=model2.predict(X_test)
pred2
sklearn.metrics.accuracy_score(Y_test,pred2)*100
classification_report(Y_test, pred2)

model3=GaussianNB()
model3.fit(X_train,Y_train)
pred3=model3.predict(X_test)
pred3
sklearn.metrics.accuracy_score(pred3,Y_test)*100
classification_report(Y_test, pred3)

model4=LogisticRegression()
model4.fit(X_train,Y_train)
pred4=model4.predict(X_test)

pred4
sklearn.metrics.accuracy_score(pred4,Y_test)*100
classification_report(Y_test, pred4)

model5=SVC()
model5.fit(X_train,Y_train)
pred5=model5.predict(X_test)
pred5
sklearn.metrics.accuracy_score(pred5,Y_test)*100
classification_report(Y_test, pred5)

model6=DecisionTreeClassifier()
model6.fit(X_train,Y_train)
pred6=model6.predict(X_test)
pred6
sklearn.metrics.accuracy_score(pred6,Y_test)*100
classification_report(Y_test, pred6)


model7=MLPClassifier()
model7.fit(X_train,Y_train)
pred7=model7.predict(X_test)
pred7
sklearn.metrics.accuracy_score(pred7,Y_test)*100
classification_report(Y_test, pred7)
