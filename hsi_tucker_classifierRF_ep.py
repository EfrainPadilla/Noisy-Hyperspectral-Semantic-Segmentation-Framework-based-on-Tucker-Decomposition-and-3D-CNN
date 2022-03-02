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

print("SNR=", SNR)
print("Alpha=", alpha)

import pathlib
root_path = str(pathlib.Path().resolve())

#Path selection, alternating between 'sourceDatasets' for original and 'noisyDatasets/xdB' for HSI with sinthetic noise
pathS= root_path + '/noisyDatasets/' + SNR + '/' + alpha + '/'
#Preprocessing - Scale and dimensionality reduction
preprocessing = "standardScaler" #Choices= ["standardScaler"], ["minmax"], ["onlyPCA"], [None]

"""## **Image selection, load and preprocessing with PCA**"""

def load_image(HSI, preprocessing, path):
  from numpy import load
  from numpy import array
  from numpy import newaxis
  from sklearn.preprocessing import MinMaxScaler, StandardScaler

  print("---The HSI selected is:", HSI, "---")
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
  print("The shape of the image is:",data.shape)
  print("The shape of the labels is:",labels.shape)
  print("Number of classes: ",num_class)
  if preprocessing != None:
    shapeor = data.shape
    data = data.reshape(-1, data.shape[-1])
    if preprocessing == "standardScaler": data = StandardScaler().fit_transform(data); print("Standard Scaler preprocessing method applied")
    elif preprocessing == "minmax": data = MinMaxScaler().fit_transform(data); print("MinMax preprocessing method applied")
    else: print("[WARNING] Not preprocessing method selected")
    data = data.reshape(shapeor)
  else: print("[WARNING] Not preprocessing method selected")
  return data, labels, num_class

(data, labels, num_class) = load_image(HSI, preprocessing, pathS)

# #Getting rank3 for tucker decomposition
# import vbmfHSI
# tensorial_bands = vbmfHSI.EVBMF(data.reshape(-1, data.shape[-1]))
# # tensorial_bands = 40
# print("The rank3 for Tucker decomposition is: ", tensorial_bands ," obtained by Variational Bayes Matrix Factorization")

tensorial_bands = 40

import tuckerHSI
#Tucker preprocessing
d1 = data.shape[0]
d2 = data.shape[1]
random_state = 12345
tucker_rank = [d1, d2, tensorial_bands]
data, tucker_factors = tuckerHSI.tucker(data,ranks=tucker_rank, init='random', tol=10e-5, random_state=random_state)
print("The new dimensions for the compressed HSI is:",data.shape ,"obtained by Tucker")




from numpy import zeros

"""## **Splitting into train and test**"""

def split_data(data, labels):
  from sklearn.model_selection import train_test_split
  # PARAMETERS--------------------------------------------------------------------------------------------------
  trainsize=0.03
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

"""# **Classifier - RF**"""

from sklearn.ensemble import RandomForestClassifier
import time

start = time.time()  # Empieza a contar el tiempo

clf = RandomForestClassifier(\
                    n_estimators=200, min_samples_split=2, \
                    max_features=40, max_depth=60) \
                    .fit(X_train, y_train)
print("\nTerminado en " + str(time.time() - start) + " segundos!\n\n")
print("Trained!")

"""Analysis of accuracy and loss during 3D CNN training"""

# import matplotlib.pyplot as plt
# # summarize history for accuracy
# plt.figure(figsize=(10, 5))
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Accuracy analysis')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='lower right')
# plt.grid()
# plt.show()
#
# # summarize history for loss
# plt.figure(figsize=(10, 5))
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Loss analysis')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper right')
# plt.grid()
# plt.show()

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

y_pred = clf.predict(X_test)

classification = classification_report(y_test, y_pred)
oa = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
each_acc, aa = AA_andEachClassAccuracy(confusion)
kappa = cohen_kappa_score(y_test, y_pred)

"""### Classification report"""

print("Classification report:")
print(classification)

"""### Accuracy Score"""

print("Accuracy Score:", oa)

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
print("Accuracy by each class:", each_acc)
print("Average accuracy", aa)

"""### Cohen’s kappa score"""

print("Cohen’s kappa score: ", kappa)
