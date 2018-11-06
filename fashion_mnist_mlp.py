# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 11:08:29 2018

@author: Yuming
"""

from __future__ import print_function

# Keras Models
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.datasets import fashion_mnist
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
import itertools
import time


start_time = time.time()


x_train, y_train, x_valid, y_valid, x_test ,y_test = [], [], [], [], [], []
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(60000,28,28)
x_test = x_test.reshape(10000,28,28)


#%%
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=1/3)

x_train = x_train.astype('float32')
x_valid = x_valid.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_valid /= 255
x_test /= 255

x_train = x_train.reshape(len(x_train),28,28)
x_test = x_test.reshape(10000,28,28)
x_valid = x_valid.reshape(len(x_valid),28,28)


y_train = keras.utils.to_categorical(y_train, 10)
y_valid = keras.utils.to_categorical(y_valid, 10)
y_test = keras.utils.to_categorical(y_test, 10)
#%% PCA
# find components
x_train = x_train.reshape(40000,784)
x_valid = x_valid.reshape(20000,784)
x_test = x_test.reshape(10000,784)

n_components = 196

print("Extracting the top %d eigenvectors from %d images" % (n_components, x_train.shape[0]))
#pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(x_train)
pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True)
pca.fit(x_train)


x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)
x_valid_pca = pca.transform(x_valid)

n_len = int(np.sqrt(n_components))
x_train = x_train_pca.reshape(len(x_train),n_len,n_len)
x_test = x_test_pca.reshape(10000,n_len,n_len)
x_valid = x_valid_pca.reshape(len(x_valid),n_len,n_len)

#%%

# Model
model = Sequential()
## input layer
model.add(Flatten(input_shape=(n_len,n_len)))

# first hidden layer
model.add(Dense(150, activation='relu', kernel_initializer = 'he_normal' ,input_shape=(28*28,)))
model.add(Dropout(0.2))
# second hidden layer
model.add(Dense(120, activation='relu',kernel_initializer = 'he_normal'))
model.add(Dropout(0.2))
# third hidden layer
model.add(Dense(90, activation='relu',kernel_initializer = 'he_normal'))
model.add(Dropout(0.2))
# third hidden layer
model.add(Dense(40, activation='relu',kernel_initializer = 'he_normal'))
model.add(Dropout(0.2))
# output layer
model.add(Dense(10, activation='softmax',kernel_initializer = 'he_normal'))


    
optim = keras.optimizers.SGD(lr=0.0005, momentum=0.975, decay=0, nesterov=True)

model.compile(loss='categorical_crossentropy',
			  optimizer=optim,
			  metrics=['accuracy'])

callbacks = [EarlyStopping(monitor='val_loss', patience=5),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

history = model.fit(x_train, y_train,
					batch_size=100,
					epochs=200,
                callbacks=callbacks,
					verbose=2,
					validation_data=(x_valid, y_valid))



# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Learning curve')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()



score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test top 1 accuracy:', score[1])
complete_time = time.time() - start_time
print(complete_time)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

test_pred = model.predict(x_test)
test_pred = np.argmax(test_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

print('Confusion Matrix')
test_confusion_matrix = confusion_matrix(y_true=y_test, y_pred=test_pred, labels=list(range(10)))  
print(test_confusion_matrix)


train_pred = model.predict(x_train)
train_pred = np.argmax(train_pred, axis=1)
y_train = np.argmax(y_train, axis=1)

print('Confusion Matrix')
train_confusion_matrix = confusion_matrix(y_true=y_train, y_pred=train_pred, labels=list(range(10)))  
print(train_confusion_matrix)

valid_pred = model.predict(x_valid)
valid_pred = np.argmax(valid_pred, axis=1)
y_valid = np.argmax(y_valid, axis=1)

print('Confusion Matrix')
valid_confusion_matrix = confusion_matrix(y_true=y_valid, y_pred=valid_pred, labels=list(range(10)))  
print(valid_confusion_matrix)

#%%

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
#%%

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(train_confusion_matrix, classes=class_names, normalize=True,
                      title='Train confusion matrix')
plt.show()

plt.figure()
plot_confusion_matrix(valid_confusion_matrix, classes=class_names, normalize=True,
                      title='Validation confusion matrix')
plt.show()

plt.figure()
plot_confusion_matrix(test_confusion_matrix, classes=class_names, normalize=True,
                      title='Test confusion matrix')
plt.show()
