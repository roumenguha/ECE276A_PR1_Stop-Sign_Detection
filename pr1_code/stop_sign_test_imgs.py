# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 11:34:48 2020

@author: roume
"""

import os, cv2
import numpy as np
import matplotlib as plot
from skimage.measure import label, regionprops
from scipy.special import expit as sigmoid # used for numerical stability

inputPath = 'C:/Users/roume/PycharmProjects/ECE276A_Project1_StopSignDetector/venv/validset'
outputPath = 'C:/Users/roume/PycharmProjects/ECE276A_Project1_StopSignDetector/venv/validset/masks2'

weights = np.array([ -262.96844267, -2173.87815346, -5326.3292497, 5020.39252529]) # trained on only the training set, 50/50 red/notRed, 500000 pixels total
weights = np.array([-1051.98, 7793.46, -17668, 7497.26]) # trained on both training and validation, used problematic pixels for notRed dataset, 600000 pixels total
weights = np.array([-1198.70681915, 2975.83862542, -19176.06813495, 9368.96722851]) # removed a third of the red pixels so the entire dataset would be evenly split 50/50

os.chdir(inputPath)
filelist = os.listdir(inputPath)

for file in filelist[:]:
    if not(file.endswith(".jpg")):
        filelist.remove(file)
    else:
        img = cv2.imread(file)
        
        x = img.flatten().reshape(img.shape[0] * img.shape[1], img.shape[2])
        
        intercept = np.ones((x.shape[0], 1))
        x = np.concatenate((intercept, x), axis = 1)
        
        mask = np.matmul(x, weights)
        mask = mask.reshape(img.shape[0], img.shape[1])
#        mask = 1.0 * (mask > 0) # comment this out if outputing; other programs read binary images as grayscale, so they appear black
        
        cv2.imshow('', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()        
        
        os.chdir(outputPath)
        cv2.imwrite(file, mask)
        
        os.chdir(inputPath)
