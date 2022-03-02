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

start = time.time()  # Empieza a contar el tiempo
dataTucker, tucker_factors = tuckerHSI.tucker(data,ranks=tucker_rank, init='random', tol=10e-5, random_state=random_state)
print("\nTucker finished in " + str(time.time() - start) + " seconds!\n\n")

print("The old dimensions of the compressed HSI is:",data.shape ,"obtained by Tucker")
print("The new dimensions for the compressed HSI is:",dataTucker.shape ,"obtained by Tucker")


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
  print("The new shape of the data is: ", patchesData.shape)
  print("The new shape of the labels is: ", patchesLabels.shape)
  return patchesData, patchesLabels.astype("int")

start = time.time()  # Empieza a contar el tiempo
(data, labelsorig) = creating_smaller_cubes(data, labels)
print("\nCubes of original data created in " + str(time.time() - start) + " seconds!\n\n")

start = time.time()  # Empieza a contar el tiempo
(dataTucker, labelstuck) = creating_smaller_cubes(dataTucker, labels)
print("\nCubes of compressed data created in " + str(time.time() - start) + " seconds!\n\n")

