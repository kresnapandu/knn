import pandas as pd

data = pd.read_csv('B:\\python\\Data\\urine_dataset.csv')
x = data.iloc[:,0:4].values
y = data.iloc[:,-1].values
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.30, random_state=10)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
k = np.arange(1,100)
akurasi = []
for i in k:
    classifier = KNeighborsClassifier(n_neighbors=i, metric='euclidean').fit(x_train, y_train)
    acc = classifier.score(x_test, y_test)
    akurasi.append(acc)

print(akurasi)
import matplotlib.pyplot as plt

plt.plot(k, akurasi)
plt.xlabel('K')
plt.ylabel('Accuration') 
plt.title('KNN Data test 30%')
plt.show()

x_pred = [[7.3,1,1,200]]
classifier = KNeighborsClassifier(n_neighbors=3, metric='euclidean').fit(x_train, y_train)
acc = classifier.score(x_test, y_test)
y_pred = classifier.predict(x_pred)
print(y_pred)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
