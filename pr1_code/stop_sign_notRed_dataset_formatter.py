'''
ECE276A WI20 HW1
Stop Sign Extractor
'''

import os, cv2
import numpy as np
import matplotlib as plot
from skimage.measure import label, regionprops

inputPath = 'C:/Users/roume/PycharmProjects/ECE276A_Project1_StopSignDetector/venv/notRedPrimeDataset'

notRedDataset = np.empty([1, 3], dtype = int)

os.chdir(inputPath)
filelist = os.listdir(inputPath)

for file in filelist[:]:
    if not(file.endswith(".jpg")):
        filelist.remove(file)
    else:
        imBGR = cv2.imread(file)
        height, width, channels = imBGR.shape
        
        notRedPixelIndices = np.where(np.all(imBGR >= [0, 0, 40], axis = -1))
        notRedPixels = imBGR[notRedPixelIndices]
        
        notRedDataset = np.append(notRedDataset, notRedPixels, axis = 0)

notRedDataset = notRedDataset[1:]

#notRedDataset[0::80]
#notRedDataset[0::170]