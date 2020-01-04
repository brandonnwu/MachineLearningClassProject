#Brandon Wu
#ITP 449 Fall '19
#Final project
# Q1

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
import numpy as np

df=pd.read_csv('winequality.csv')
X=df.iloc[:,0:-1]
y=df.iloc[:,-1]

print(X,y)
#A. Standardizing all variables except quality
scaler = StandardScaler()
scaler.fit(X) #standardize all variables other than species
X= pd.DataFrame(scaler.transform(X),columns=X.columns)

#B. splitting the dataset into a training dataset and a testing dataset (70/30)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=2019, stratify=y)


#C. Iterating K ranging from 1 to 10 and plotting accruacy of training and testing datasets
neighbors = np.arange(1,10)
train_accuracy = np.empty(9)
test_accuracy = np.empty(9)

for k in range(1,10):
    print(k)
    knn= KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred =knn.predict(X_test)
    cf=metrics.confusion_matrix(y_test,y_pred)
    # print(cf)
    train_accuracy[k-1] = knn.score(X_train,y_train)
    test_accuracy[k-1] = knn.score(X_test,y_test)

plt.figure(2)
plt.plot(neighbors,test_accuracy,label= 'Testing Accuracy')
plt.plot(neighbors,train_accuracy,label= 'Training Accuracy')
plt.legend()
plt.xlabel('number of K')
plt.ylabel('Accuracy')
plt.show()


#E Generating the predictions for the test partition with the value of K of 1
knn1= KNeighborsClassifier(n_neighbors=1)
knn1.fit(X_train,y_train)
y_pred1 =knn1.predict(X_test)

print(y_pred1)

#F the accuracy of the model
test_acc = accuracy_score(y_pred, y_test)
print("Accuracy on the test partition: ", test_acc)
