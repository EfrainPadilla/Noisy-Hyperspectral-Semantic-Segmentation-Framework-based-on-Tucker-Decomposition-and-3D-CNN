import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import time
start = time.time()
from datetime import datetime
print(datetime.now())

import sys
"""## **Parameters**"""

HSI = str(sys.argv[1])
alpha = str(sys.argv[2])
SNR = str(sys.argv[3])

import pathlib
root_path = str(pathlib.Path().resolve())

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

from numpy import newaxis
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import time
import pathlib
path = str(pathlib.Path().resolve()) + '/TrainedModels/'+HSI+'/'

"""## **Reports**"""

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
from numpy import argmax

def AA_andEachClassAccuracy(confusion_matrix):
  import numpy as np
  counter = confusion_matrix.shape[0]
  list_diag = np.diag(confusion_matrix)
  list_raw_sum = np.sum(confusion_matrix, axis=1)
  each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
  average_acc = np.mean(each_acc)
  return each_acc, average_acc

#Here begins the prediction
clf = load_model(path)
pathS= root_path + '/noisyDatasets/' + SNR + '/' + alpha + '/'
print("Loading "+ HSI+ " HSI at: "+pathS)
(data, labels, num_class) = load_image(HSI, preprocessing, pathS)
(data, labels) = creating_smaller_cubes(data, labels)
data = data[labels!=0]
labels = labels[labels!=0] - 1
data  = data[..., newaxis]



y_pred = argmax(clf.predict(data), axis=1)
del clf
K.clear_session()

# classification = classification_report(labels, y_pred)
# oa = accuracy_score(labels, y_pred)
# confusion = confusion_matrix(labels, y_pred)
# each_acc, aa = AA_andEachClassAccuracy(confusion)
kappa = cohen_kappa_score(labels, y_pred)

"""### Classification report"""

# print("Classification report:")
# print(classification)

"""### Accuracy Score"""

# print("Accuracy Score:", oa)

"""### Confusion Matrix"""

# def plot_confusion_matrix(cm,
#                           target_names,
#                           title='Confusion matrix',
#                           cmap=None,
#                           normalize=True):
#   import matplotlib.pyplot as plt
#   import itertools
#   import numpy as np
#
#   accuracy = np.trace(cm) / float(np.sum(cm))
#   misclass = 1 - accuracy
#
#   if cmap is None:
#       cmap = plt.get_cmap('Blues')
#
#   plt.figure(figsize=(20, 14))
#   plt.imshow(cm, interpolation='nearest', cmap=cmap)
#   plt.title(title)
#   plt.colorbar()
#
#   if target_names is not None:
#       tick_marks = np.arange(len(target_names))
#       plt.xticks(tick_marks, target_names, rotation=45)
#       plt.yticks(tick_marks, target_names)
#
#   if normalize:
#       cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#
#
#   thresh = cm.max() / 1.5 if normalize else cm.max() / 2
#   for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#       if normalize:
#           plt.text(j, i, "{:0.4f}".format(cm[i, j]),
#                     horizontalalignment="center",
#                     color="white" if cm[i, j] > thresh else "black")
#       else:
#           plt.text(j, i, "{:,}".format(cm[i, j]),
#                     horizontalalignment="center",
#                     color="white" if cm[i, j] > thresh else "black")
#
#
#   plt.tight_layout()
#   plt.ylabel('True label')
#   plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
#   plt.show()
#
# if HSI == "sentinel2":
#   names = ['Soil', 'Shadow', 'Cloud', 'Vegetation', 'Water']
# if HSI == 'paviaU':
#   names = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil', 'Bitumen', 'Self-Blocking Bricks', 'Shadows']
# if HSI == 'indianPines':
#   names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn', 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed', 'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill', 'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives', 'Stone-Steel-Towers']
# if HSI == 'salinas':
#   names = ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow', 'Fallow_smooth', 'Stubble', 'Celery', 'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds', 'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk	', 'Vinyard_untrained', 'Vinyard_vertical_trellis']
#
# plot_confusion_matrix(cm           = confusion,
#                       normalize    = False,
#                       target_names = names,
#                       title        = "Confusion Matrix")

"""### Accuracy analysis"""

from numpy import set_printoptions
set_printoptions(precision=3)
# print("Accuracy by each class:", each_acc)
# print("Average accuracy", aa)

"""### Cohen’s kappa score"""

print("Cohen’s kappa score: ", kappa)
