import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

mnist = tf.keras.datasets.mnist

(x_traino, y_train), (x_testo, y_test) = mnist.load_data()#get train and test data
x_train = np.reshape(x_traino,(60000, 28*28))
x_test = np.reshape(x_testo, (10000, 28*28))
x_train, x_test = x_train/255.0, x_test/255.0
logreg = LogisticRegression(solver='saga', multi_class='multinomial', max_iter=100, verbose=2)

logreg.fit(x_train,y_train)#start training
y_answer = logreg.predict(x_test)#start testing

correct = 0#calculate accuracy
for i in range(len(y_test)):
    if y_answer[i]==y_test[i]:
        correct+=1
accuracy = correct/len(y_test)
print("ACCURACY IS: ", accuracy)#show accuracy

cm = confusion_matrix(y_test, y_answer)#produce confusion matrix
score = logreg.score(x_test, y_test)
#show confusion matrix in plot
plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r');##
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Confusion Matrix' 
plt.title(all_sample_title, size = 15);
