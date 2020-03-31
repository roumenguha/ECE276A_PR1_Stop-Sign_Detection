'''
ECE276A WI20 HW1
Stop Sign Extractor
'''

import os, cv2
import numpy as np
import matplotlib as plot
from skimage.measure import label, regionprops

inputPath = 'C:/Users/roume/PycharmProjects/ECE276A_Project1_StopSignDetector/venv/validset'
positiveOutputPath = 'C:/Users/roume/PycharmProjects/ECE276A_Project1_StopSignDetector/venv/validset/RedPositive'
negativeOutputPath = 'C:/Users/roume/PycharmProjects/ECE276A_Project1_StopSignDetector/venv/validset/RedNegative'

os.chdir(inputPath)
filelist = os.listdir(inputPath)

lower_red_HSV = np.array([20,150,50])
upper_red_HSV = np.array([255,255,220])

for file in filelist[:]:
    if not(file.endswith(".jpg")):
        filelist.remove(file)
    else:
        imBGR = cv2.imread(file)
        imRGB = cv2.cvtColor(imBGR, cv2.COLOR_BGR2RGB)
        imHSV = cv2.cvtColor(imRGB, cv2.COLOR_BGR2HSV)

        os.chdir(positiveOutputPath)
        posMask = cv2.inRange(imHSV, lower_red_HSV, upper_red_HSV)
        posRes = cv2.bitwise_and(imBGR, imBGR, mask = posMask)
        cv2.imwrite(file, posRes)
        
        os.chdir(negativeOutputPath)
        negMask = cv2.bitwise_not(posMask)
        negRes = cv2.bitwise_and(imBGR, imBGR, mask = negMask)
        cv2.imwrite(file, negRes)
        
        os.chdir(inputPath)
        
        

