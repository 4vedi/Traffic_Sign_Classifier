import matplotlib.pyplot as plt 
import numpy as np 
import tensorflow as tf 
import pandas as pd 
import seaborn  as sns 
import pickle 
import random

# IMPORTING GTSRB DATASET
with open("./traffic-signs-data/train.p", mode = 'rb') as training_data:
	train = pickle.load(training_data)
with open("./traffic-signs-data/valid.p", mode = 'rb') as validation_data:
	valid = pickle.load(validation_data)
with open("./traffic-signs-data/test.p", mode = 'rb') as testing_data:
	test = pickle.load(testing_data)

X_train, y_train = train['features'], train['labels']
X_test, y_test = train['features'], train['labels']
X_validate, y_validate = valid['features'], valid['labels']

# CONVERT IMAGES AND NORMALIZATION

from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)

X_train_gray = np.sum(X_train/3, axis = 3, keepdims = True)

X_test_gray = np.sum(X_test/3, axis = 3, keepdims = True)
X_validate_gray = np.sum(X_validate/3, axis = 3, keepdims = True)

X_train_gray_norm = (X_train_gray - 128)/128
X_test_gray_norm = (X_test_gray - 128)/128
X_validate_gray_norm = (X_validate_gray - 128)/128

# BUILDING NERUAL NETWORK 

from tensorflow.keras import datasets, layers, models
CNN = models.Sequential()
CNN.add(layers.Conv2D(6, (5,5), activation = 'relu', input_shape = (32,32,1)))
CNN.add(layers.AveragePooling2D())
CNN.add(layers.Dropout(0.2))
CNN.add(layers.Conv2D(16, (5,5), activation = 'relu'))
CNN.add(layers.Flatten())
CNN.add(layers.Dense((120), activation = 'relu'))
CNN.add(layers.Dense((84), activation = 'relu'))
CNN.add(layers.Dense((43), activation = 'softmax'))


# COMPILE AND TRAIN

CNN.compile(optimizer = 'Adam', loss = "sparse_categorical_crossentropy", metrics = ['accuracy'])
history = CNN.fit(X_train_gray_norm, y_train, batch_size = 500, epochs = 25, verbose = 1, validation_data = (X_validate_gray_norm, y_validate))

score = CNN.evaluate(X_test_gray_norm, y_test)
print('Test accuracy: {}'.format(score[1]))

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

#epochs = range(len(accuracy))
#plt.plot(epochs, loss, 'ro', label = 'Training loss')
#plt.plot(epochs, val_loss, 'r', label = 'Validation loss')
#plt.title('Training and validation loss')

predicted_classes = CNN.predict_classes(X_test_gray_norm)
y_true = y_test
from sklearn.metrics import confusion_matrix
cm = confusion_matric(y_true, predicted_classes)
plt.fihure(figsize = (25,25))
sns.heatmap(cm, annot = True)

L = 5
W = 5
fig, axes = plt.subplots(L,W, figsize=(12,12))
axes = axes.ravel()
for i in np.arange(0,L*W):
	axes[i].imshow(X_test[i])
	axes[i].set_title('Prediction = {}\n True = {}'.format(predicted_classes[i]), y_true[i]))
	axes[i].axis('off')
plt.subplots_adjust(wspace - 1)

