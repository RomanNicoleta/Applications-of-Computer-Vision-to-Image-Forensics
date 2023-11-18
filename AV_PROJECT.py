# -*- coding: utf-8 -*-
"""
Title: Fake Image Detection with ELA and CNN

Purpose: Artificial Vision Course Project
Practical implementation of "Applications of Computer Vision to Image Forensics"

@author: Agus Gunawan, Holy Lovenia, Adrian Hartanto Pramudita

Source: https://github.com/agusgun/FakeImageDetector/blob/master/fake-image-detection.ipynb

Commits on Oct 14, 2018

Modified by: Nicoleta Roman 
Last update: May 07, 2023
"""
#%% Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

sns.set(style='white', context='notebook', palette='deep')

#%% Import more packages
from PIL import Image
import os
from pylab import *
import re
from PIL import Image, ImageChops, ImageEnhance
from PIL import ImageFilter


#%% Get img list
def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')]

#%% Define ELA 
def convert_to_ela_image(path, quality):
    
    filename = path
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ['.jpg', '.jpeg', '.png']:
        print(f"{filename} is not an image file.")
        return None
   
    try:
        image = Image.open(filename).convert('RGB')
        if image is None:
            print(f"{filename} could not be opened as an image.")
            return None
    except:
        print(f"{filename} is not an image file.")
        return None
    
    
    resaved_filename = filename.split('.')[0] + '.resaved.jpg'
    ELA_filename = filename.split('.')[0] + '.ela.png'
    
    im = Image.open(filename).convert('RGB')
    im.save(resaved_filename, 'JPEG', quality=quality)
    resaved_im = Image.open(resaved_filename)
    
    ela_im = ImageChops.difference(im, resaved_im)
    
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
    
    return ela_im



#%% Make a CSV file

import os
import csv

directory = 'D:\dataset\CASIA2'

with open('data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['file_path', 'label'])
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            
            if file.startswith('Au'):
                label = 0
            else:
                label = 1
                
            writer.writerow([file_path, label])

#%% Apply ELA

import cv2
import numpy as np
import pandas as pd



dataset = pd.read_csv('data.csv')
X = []
Y = []

    
for index, row in dataset.iterrows():
    ela_image = convert_to_ela_image(row['file_path'], 90)
    if ela_image is None:
        continue
    resized_image = ela_image.resize((128, 128))
    X.append(array(resized_image).flatten() / 255.0)
    Y.append(row['label'])

#%% Pre-processing 
# NORMALIZATION 

X = np.array(X)
Y = to_categorical(Y, 2)

# RESHAPE X
X = X.reshape(-1, 128, 128, 3)

# TRAIN TEST SPLIT
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)
#%% CNN
# CNN BUILDING

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'valid', 
                 activation ='relu', input_shape = (128,128,3)))
print("Input: ", model.input_shape)
print("Output: ", model.output_shape)

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'valid', 
                 activation ='relu'))
print("Input: ", model.input_shape)
print("Output: ", model.output_shape)

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))
print("Input: ", model.input_shape)
print("Output: ", model.output_shape)

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(2, activation = "softmax"))

#%%
model.summary()


#%%
optimizer = RMSprop(lr=0.0005, rho=0.9, epsilon=1e-08, decay=0.0)


#%%
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


#%%
early_stopping = EarlyStopping(monitor='val_accuracy',
                              min_delta=0,
                              patience=30,
                              verbose=0, mode='auto')

#%% Training
# MODEL TRAINING 
epochs = 20
batch_size = 50

history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, 
                    validation_split=0.2, verbose=2, callbacks=[early_stopping]) # using validation_split to split the training data into train and validation internally

#%% Evaluation
# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
test_f1 = f1_score(Y_test.argmax(axis=1), Y_pred.argmax(axis=1), average='weighted')
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)
print('Test F1 score:', test_f1)


#%% Metrics
# METRICS
import matplotlib.pyplot as plt

# Plot the training and testing loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()

# Plot the training and testing accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.show()


#%% Confusion matrix
# CONFUSSION MATRIX 

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
# Predict the values from the test dataset
Y_pred = model.predict(X_test)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_test,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(2))

