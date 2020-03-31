'''
ECE276A WI20 HW1
Stop Sign Extractor
'''

import os, cv2
import numpy as np
import matplotlib as plot
from skimage.measure import label, regionprops

inputPath = 'C:/Users/roume/PycharmProjects/ECE276A_Project1_StopSignDetector/venv/validset'
positiveInputPath = 'C:/Users/roume/PycharmProjects/ECE276A_Project1_StopSignDetector/venv/validset/RedPositive'
negativeInputPath = 'C:/Users/roume/PycharmProjects/ECE276A_Project1_StopSignDetector/venv/validset/RedNegative'

redDataset = np.empty([1, 3], dtype = int)

os.chdir(positiveInputPath)
filelist = os.listdir(positiveInputPath)

for file in filelist[:]:
    if not(file.endswith(".jpg")):
        filelist.remove(file)
    else:
        imBGR = cv2.imread(file)
        height, width, channels = imBGR.shape
        
        redPixelIndices = np.where(np.all(imBGR >= [0, 0, 40], axis = -1))
        redPixels = imBGR[redPixelIndices]
        
        redDataset = np.append(redDataset, redPixels, axis = 0)

redDataset = redDataset[1:]