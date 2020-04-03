from sklearn.decomposition import PCA
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

X = np.zeros(shape=(1,112*92))
y = []
path = "./att_faces_10/"
picAmount = 0
for root, dirs, files in os.walk(path, topdown=False):#read all images and their labels
    for name in dirs:
        for root, dirs, files in os.walk(path+name+"/", topdown=False):
            for pic in files:
                label = int(name[1:])
                y.append(label)
                im = Image.open(path+name+"/"+pic);
                X = np.r_[X, np.array(im).reshape(1,-1)]
y = np.array(y).reshape((-1,1))
X = X[1:,:] #delete the first row which is all zero
#randomly choose 20% to form test set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
sum = 0
for i in range (0,20):
    d = 30
    pca0 = PCA(n_components=d)
    pca0_operator = pca0.fit(x_train) #each row is one training image
    x_train = pca0_operator.transform(x_train)
    x_test = pca0_operator.transform(x_test)
    #use knn to classify
    knn = KNeighborsClassifier(n_neighbors = 10)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    sum+=accuracy_score(y_test, y_pred)

print ('accuracy', sum/20)