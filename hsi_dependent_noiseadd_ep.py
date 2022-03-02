import time
start = time.time()
from datetime import datetime
print(datetime.now())

import sys
"""## **Parameters**"""
#Choices= [paviaU], [indianPines], [salinas]
HSI = str(sys.argv[1])
SNR = int(sys.argv[3])
alpha = float(sys.argv[2])
print("SNR=", SNR)
print("Alpha=", alpha)

def load_image(HSI):
  from numpy import load
  import pathlib
  root_path = str(pathlib.Path().resolve())
  path= root_path + '/sourceDatasets/'

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
  return data, labels, num_class, data_name, labels_name

(data, labels, num_class, data_name, labels_name) = load_image(HSI)

#Check for negative values, if is true, the pixel will be 0
(I1,I2,I3) = data.shape

from numpy import min

if min(data) < 0 :
  data[data < 0] = 0
  print("Found negatives in source values")
  if min(data) < 0 :
      print("Correction of negatives failed")
  else:
      print("Correction of negatives done!")

"""# **Noise generation**

## **Functions**
"""

#Frobenius Norm
def fro_norm(X):
  from math import sqrt as sqrt_float
  from numpy import square as square_tensor
  from numpy import sum as summatory
  return sqrt_float(summatory(square_tensor(X)))

#Power of a signal
def power_signal(X):
  from numpy import square as square_tensor
  from numpy import sum as summatory
  (I1, I2, I3) = X.shape
  return (1/(I1*I2*I3))*summatory(square_tensor(X))

"""##**Packages**"""

#Math
from numpy import ones
from math import sqrt as sqrt_float
from numpy import multiply as HadamardProduct
from numpy import sqrt as sqrt_tensor
from numpy import zeros
from numpy import sum as summatory
from numpy import mean
#Graph
from numpy import arange
# import matplotlib.pyplot as plt
#Normal Distribution generator
from numpy.random import normal
#Quantization
from numpy import rint

"""## **Computing average variances**"""

(I1, I2, I3) = data.shape
print(data.shape)

#Variance of the noise in "general"
average_variance_of_noise = power_signal(data) * ( 10 ** (-SNR / 10))
print("The average variance of the noise is: " ,average_variance_of_noise)

#Variance of the signal-dependent noise
total_variance_SD = (average_variance_of_noise*alpha)/(1 + alpha)
print("The total variance of the dependent noise is: ",total_variance_SD)

#Variance of the signal-independent noise
total_variance_SI = average_variance_of_noise / (alpha + 1)
print("The total variance of the independent noise is: ",total_variance_SI)

"""## **Computing variances per band**"""

#Variance of dependent Noise
variances_SD = zeros((I3))

for i3 in range(0, I3):
  variances_SD[i3] = (total_variance_SD*I1*I2) / (summatory(data[:,:,i3]))

print("The variances of dependent noise per band are:")
print(variances_SD)

#Plot of dependent Noise
# y_pos = arange(1,I3+1)
# fig, ax = plt.subplots(figsize=(20,3))
# plt.bar(y_pos, variances_SD, align='center', alpha=1)
# plt.style.use(['bmh'])
#
# plt.xlim(0, I3+1)
# plt.ylabel('Variance of Dependent Noise')
# plt.xlabel('Bands')
# plt.title('Variances per band of Dependent Noise')
# plt.show()

#Variance of independent Noise

variance_SI = total_variance_SI
print("The variance of the independent noise is: ",variance_SI)

"""## **Gaussian Noise Generation**"""

#Creating tensors
RV_SD = zeros((I1, I2, I3))
dependentNoise = zeros((I1, I2, I3))
independentNoise = zeros((I1, I2, I3))
#Generating signal-dependent noise
for i3 in range(0, I3):
  RV_SD[:,:,i3] = normal(loc=0, scale=sqrt_float(variances_SD[i3]), size=(I1,I2))

dependentNoise = HadamardProduct(sqrt_tensor(data), RV_SD)

#Quantization of dependent noise
dependentNoise = rint(dependentNoise)

#Generating signal-independent noise
independentNoise = normal(loc=0, scale=sqrt_float(variance_SI), size=(I1,I2,I3))

#Quantization of independent noise
independentNoise = rint(independentNoise)


noise = dependentNoise + independentNoise

print(noise.shape)

"""## **Additive Noise**"""

noisy_data = data + noise

#Check for negative values, if is true, te pixel will be 0
minimum_of_data = min(noisy_data)
if minimum_of_data < 0 :
  noisy_data[noisy_data < 0] = 0
  print("Found negatives in noisy data")
  if min(data) < 0 :
      print("Correction of negatives failed")
  else:
      print("Correction of negatives done!")

print("Noise added to HSI!")

"""## **SNR Analysis**"""

#Signal-to-Noise Ratio
def snr(X, N):
  from math import log10
  from numpy import square as square_tensor
  #print("Frobenius norm of data", fro_norm(X))
  #print("Frobenius norm of noise", fro_norm(N))
  return 10 * log10( square_tensor(fro_norm(X)) / square_tensor(fro_norm(N)))

snr = snr(data, noise)
print("The SNR of the Noisy data is, SNR = ", snr)

"""## **Mean Squared Error**"""

def mse(H, X):
  from numpy import square, mean
  return (square(H - X)).mean(axis=None)

mse = mse(noisy_data, data)
print("The MSE between the noisy and clean data is = ", mse)

"""## **Visualization**"""

# #Set de band to analyze
# band=30
#
# from numpy import max, min
#
# ori = noisy_data[:,:,band]
# img = data[:,:,band]
# noi = noise[:,:,band]
#
# fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(22, 11))
#
# ax0.imshow(ori, cmap='gist_gray', vmin=None, vmax=None)
# ax0.set_title("Noisy HSI")
# ax1.imshow(img, cmap='gist_gray', vmin=None, vmax=None)
# ax1.set_title("Clean HSI")
# ax2.imshow(noi, cmap='hsv', vmin=None, vmax=None)
# ax2.set_title("Noise")
#
# plt.style.use(['bmh'])
#
# fig.tight_layout()
#
# ori = noisy_data[:,:,band]
# img = data[:,:,band]
# noi = noise[:,:,band]
# depnoi = dependentNoise[:,:,band]
# indpnoi = independentNoise[:,:,band]
#
# fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=5, figsize=(22, 11))
#
# ax0.imshow(ori, cmap='gist_gray', vmin=None, vmax=None)
# ax0.set_title("Noisy HSI")
# ax1.imshow(img, cmap='gist_gray', vmin=None, vmax=None)
# ax1.set_title("Clean HSI")
# ax2.imshow(noi, cmap='hsv', vmin=None, vmax=None)
# ax2.set_title("Noise")
# ax3.imshow(depnoi, cmap='hsv', vmin=None, vmax=None)
# ax3.set_title("Signal-Dependent Noise")
# ax4.imshow(indpnoi, cmap='hsv', vmin=None, vmax=None)
# ax4.set_title("Signal-Independent Noise")
#
# fig.tight_layout()

"""#  **Saving Noisy Data**"""
noisyfolder = str(int(SNR)) + 'dB/'
alphafolder = 'alpha-' + str(alpha) + '/'

from numpy import save #change the folder at the end of the corresponding dB
import pathlib
root_path = str(pathlib.Path().resolve())
pathsave= root_path + '/noisyDatasets/'
pathsave = pathsave + noisyfolder + alphafolder
save( pathsave + data_name , noisy_data)
save( pathsave + labels_name , labels)

print("Data saved in the following path:", pathsave + data_name)
print("Labels saved in the following path:", pathsave + labels_name)

end = time.time()
print(f"Runtime of the program is {end - start} seconds")
print("\n -------------------------------------------------------------------------------------------------------\n")
