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

(data, _, _) = load_image(HSI, preprocessing, pathS)

data = data.astype('float64')

print("----Relative error of Tucker Decomposition Reconstruction for each band configuration----")
import tuckerHSI
import tensorly as tl
from numpy.linalg import norm
#Tucker preprocessing
d1 = data.shape[0]
d2 = data.shape[1]
random_state = 12345
data_matrix = data.reshape(-1, data.shape[-1])

print("Start:", data.shape[2])

for tensorial_bands in range(data.shape[2], 0 , -1):
    tucker_rank = [d1, d2, tensorial_bands]
    tucker_core, tucker_factors = tuckerHSI.tucker(data,ranks=tucker_rank, init='random', tol=10e-5, random_state=random_state)

    tucker_reconstruction = tl.tucker_to_tensor((tucker_core, tucker_factors)).astype('float64')

    tucker_reconstruction_matrix = tucker_reconstruction.reshape(-1, data.shape[-1])
    error = (norm(data_matrix - tucker_reconstruction_matrix, ord='fro') ** 2) / (norm(data_matrix, ord='fro') ** 2)

    # print("Tensorial Bands:", tensorial_bands, "Error =", f'{error:.10f}')
    print(f'{error:.10f}')
