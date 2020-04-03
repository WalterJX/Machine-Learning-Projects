import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 

mnist = tf.keras.datasets.mnist

(x_traino, y_train), (x_testo, y_test) = mnist.load_data()#get train and test data
x_train = np.reshape(x_traino,(60000, 28*28))
x_test = np.reshape(x_testo, (10000, 28*28))
x_train, x_test = x_train/255.0, x_test/255.0

model = Sequential()
model.add(Dense(512,input_dim=784, activation='relu'))#add 2 layers in the neural network
model.add(Dense(10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['accuracy'])#use cross entropy loss function
model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=2)#train
predictions = model.predict(x_test)#test

predictions = np.rint(predictions)#let the result change to integer
pre_data = [np.argmax(x)for x in predictions]#change one hot data to array, in order to draw confusion matrix
pre_data = np.array(pre_data)

num_correct=0#calculate accuracy
for i in range(len(pre_data)):
    if(pre_data[i]==y_test[i]):
        num_correct+=1
accuracy = num_correct/len(pre_data)
cm = confusion_matrix(y_test, pre_data, labels=[0,1,2,3,4,5,6,7,8,9])#draw confusion matrix

print(cm)
print("Test accuracy is: ", accuracy)