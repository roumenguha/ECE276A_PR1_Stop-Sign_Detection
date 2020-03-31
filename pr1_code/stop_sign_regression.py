'''
ECE276A WI20 HW1
Stop Sign Extractor
'''

import os
import numpy as np
from scipy.special import expit as sigmoid # used for numerical stability

inputPath = 'C:/Users/roume/PycharmProjects/ECE276A_Project1_StopSignDetector/venv/Scripts'
NUM_ITER = 500
subsampling_factor = 10

###############################################################################
# HELPER FUNCTIONS
def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

###############################################################################
# VARIABLE SETUP

# Set directory
os.chdir(inputPath)

# Load datasets and form labels
red = np.load('redPrimeDataset.npy')[0::2] # subsample relative to notRed dataset to obtain almost equal split
redLabels = np.ones((red.shape[0], 1))

notRed = np.load('notRedPrimeDataset.npy')
notRedLabels = -1 * np.ones((notRed.shape[0], 1))

X = np.append(red, notRed, axis = 0)
y = np.append(redLabels, notRedLabels, axis = 0)

# Subsample datasets so we do less work
X = X[0::subsampling_factor] / 255 # Normalize channel values to be between 0 and 1
y = y[0::subsampling_factor]

# Create extra dimension to use as 'slack' or 'bias' term
intercept = np.ones((X.shape[0], 1))
X = np.concatenate((intercept, X), axis = 1)

# Initialize weights
omega = np.zeros(X.shape[1])

# Initialize record of loss
losses = np.zeros(NUM_ITER)

###############################################################################
# LOGISTIC REGRESSION
for i in range(NUM_ITER):
    
    # learning rate
    if (i < 200):
        alpha = 0.1
    elif (i < 400):
        alpha = 0.05
    else:
        alpha = 0.01
    
    gradient = 0
    for j in range(X.shape[0]):
        product = y[j] * np.dot(X[j, :], omega)
        gradient += y[j] * X[j, :] * (1 - sigmoid(product))
        losses[i] += np.log(1 + np.exp(-1 * sigmoid(product))) / X.shape[0] 
    
    old_omega = omega
    omega += alpha * gradient
        
    if (i % 10 == 0):
        print('at iteration i =', i)
        print('omega =', omega)
        print('loss =', losses[i])
        print()
