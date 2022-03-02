import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import time
start = time.time()
from datetime import datetime
print(datetime.now())

import sys
"""## **Parameters**"""

HSI = str(sys.argv[1])

import pathlib
root_path = str(pathlib.Path().resolve())

#Path selection, alternating between 'sourceDatasets' for original and 'noisyDatasets/xdB' for HSI with sinthetic noise
pathS= root_path + '/sourceDatasets/'
#Preprocessing - Scale and dimensionality reduction
preprocessing = "standardScaler" #Choices= ["standardScaler"], ["minmax"], ["onlyPCA"], [None]

"""## **Image selection, load and preprocessing with PCA**"""

def load_image(HSI, preprocessing, path):
  from numpy import load
  from numpy import array
  from numpy import newaxis
  from sklearn.preprocessing import MinMaxScaler, StandardScaler

  #print("---The HSI selected is:", HSI, "---")
  if HSI == 'paviaU': #For Pavia university data
    data_name = 'paviaU.npy'
    labels_name = 'paviaU_gt.npy'
    num_class = 9
  if HSI == 'indianPines': #For Indian Pines data
    data_name = 'indian_pines_corrected.npy'
    labels_name = 'indian_pines_gt.npy'
    num_class = 16
  if HSI == 'salinas': #For salinas data
    data_name = 'salinas_corrected.npy'
    labels_name = 'salinas_gt.npy'
    num_class = 16

  data = load(path+data_name)
  labels = load(path+labels_name)
  # print("The shape of the image is:",data.shape)
  # print("The shape of the labels is:",labels.shape)
  # print("Number of classes: ",num_class)
  if preprocessing != None:
    shapeor = data.shape
    data = data.reshape(-1, data.shape[-1])
    if preprocessing == "standardScaler": data = StandardScaler().fit_transform(data); #print("Standard Scaler preprocessing method applied")
    elif preprocessing == "minmax": data = MinMaxScaler().fit_transform(data); print("MinMax preprocessing method applied")
    else: print("[WARNING] Not preprocessing method selected")
    data = data.reshape(shapeor)
  else: print("[WARNING] Not preprocessing method selected")
  return data, labels, num_class

(data, labels, num_class) = load_image(HSI, preprocessing, pathS)

"""## **Creating smaller cubes**"""

from numpy import zeros

def padWithZeros(X, margin=2):
  newX = zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
  x_offset = margin
  y_offset = margin
  newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
  return newX

def creating_smaller_cubes(data, labels):
  # PARAMETERS--------------------------------------------------------------------------------------------------
  windowSize = 19 #19 from the original papere
  removeZeroLabels = None
  #-------------------------------------------------------------------------------------------------------------
  margin = int((windowSize - 1) / 2)
  zeroPaddedX = padWithZeros(data, margin=margin)
  patchesData = zeros((data.shape[0] * data.shape[1], windowSize, windowSize, data.shape[2]))
  patchesLabels = zeros((data.shape[0] * data.shape[1]))
  patchIndex = 0
  for r in range(margin, zeroPaddedX.shape[0] - margin):
      for c in range(margin, zeroPaddedX.shape[1] - margin):
          patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
          patchesData[patchIndex, :, :, :] = patch
          patchesLabels[patchIndex] = labels[r-margin, c-margin]
          patchIndex = patchIndex + 1
  if removeZeroLabels:
      patchesData = patchesData[patchesLabels>0,:,:,:]
      patchesLabels = patchesLabels[patchesLabels>0]
      patchesLabels -= 1
  # print("The new shape of the data is: ", patchesData.shape)
  # print("The new shape of the labels is: ", patchesLabels.shape)
  return patchesData, patchesLabels.astype("int")

(data, labels) = creating_smaller_cubes(data, labels)

"""## **Splitting into train and test**"""

def split_data(data, labels):
  from sklearn.model_selection import train_test_split
  # PARAMETERS--------------------------------------------------------------------------------------------------
  trainsize=0.15
  testsize=1.0-trainsize
  rand_state=True

  data = data[labels!=0]
  labels = labels[labels!=0] - 1
  return train_test_split(data, labels, test_size=testsize, stratify=labels, random_state=rand_state)


(X_train, X_test, y_train, y_test) = split_data(data, labels)
del data; del labels
print("The data shape for train is:", X_train.shape)
print("The labels shape for train is:", y_train.shape)
print("The data shape for test is:", X_test.shape)
print("The labels shape for test is:", y_test.shape)

"""# **Classifier - 3D CNN**

## **Model**
"""

from numpy import newaxis

X_test  = X_test[..., newaxis]
X_train = X_train[..., newaxis]
shapeinput = X_train.shape[1:]
w_decay=0
lr=1e-3

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, BatchNormalization, Conv3D,  Dense, Flatten, MaxPooling3D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

clf = Sequential()
clf.add(Conv3D(32, kernel_size=(5, 5, 24), input_shape=shapeinput))
clf.add(BatchNormalization())
clf.add(Activation('relu'))
clf.add(Conv3D(64, (5, 5, 16)))
clf.add(BatchNormalization())
clf.add(Activation('relu'))
clf.add(MaxPooling3D(pool_size=(2, 2, 1)))
clf.add(Flatten())
clf.add(Dense(300, kernel_regularizer=regularizers.l2(w_decay)))
clf.add(BatchNormalization())
clf.add(Activation('relu'))
clf.add(Dense(num_class, activation='softmax'))
clf.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
clf.summary()

"""## **Parameters**"""

batch_size = 100
epochs = 40

"""## **Training**"""

from tensorflow.keras.utils import to_categorical as keras_to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import time

start = time.time()  # Empieza a contar el tiempo

history=clf.fit(X_train, keras_to_categorical(y_train, num_class),
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose='store_true',
                 validation_data=(X_test, keras_to_categorical(y_test, num_class)),
                 callbacks = [ModelCheckpoint("/tmp/best_model.h5", monitor='val_accuracy', verbose=2, save_best_only=True)])
del clf; K.clear_session();
clf = load_model("/tmp/best_model.h5")
print("PARAMETERS", clf.count_params())
print("\nTerminado en " + str(time.time() - start) + " segundos!\n\n")

import pathlib
path = str(pathlib.Path().resolve()) + '/TrainedModels/'+HSI+'/'
clf.save(path)