import time
start = time.time()
from datetime import datetime
print(datetime.now())

import sys
"""## **Parameters**"""

HSI = str(sys.argv[1])
alpha = str(sys.argv[2])
SNR = str(sys.argv[3])

print("HSI=", HSI)
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

  # print("---The HSI selected is:", HSI, "---")
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

  return data, labels, num_class

(data, labels, num_class) = load_image(HSI, preprocessing, pathS)

#Getting rank3 for tucker decomposition
import vbmfHSI
tensorial_bands = vbmfHSI.EVBMF(data.reshape(-1, data.shape[-1]))
print("The rank3 for Tucker decomposition is: ", tensorial_bands ," obtained by Variational Bayes Matrix Factorization")
print(tensorial_bands)
#this is a test
