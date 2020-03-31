'''
ECE276A WI20 HW1
Stop Sign Extractor
'''

import os, cv2
import numpy as np
import matplotlib as plot
from skimage.measure import label, regionprops

inputPath = 'C:/Users/roume/OneDrive/Documents/Classes/5 - Masters/ECE 276A/Homework/2020_ECE276A_PR1/hw1_starter_code/trainset/'
positiveOutputPath = 'C:/Users/roume/OneDrive/Documents/Classes/5 - Masters/ECE 276A/Homework/2020_ECE276A_PR1/hw1_starter_code/trainset/RedPositive'
negativeOutputPath = 'C:/Users/roume/OneDrive/Documents/Classes/5 - Masters/ECE 276A/Homework/2020_ECE276A_PR1/hw1_starter_code/trainset/RedNegative'

os.chdir(inputPath)
file = '41.jpg'

lower_red_HSV = np.array([20,150,50])
upper_red_HSV = np.array([255,255,240])

imBGR = cv2.imread(file)
imRGB = cv2.cvtColor(imBGR, cv2.COLOR_BGR2RGB)
imHSV = cv2.cvtColor(imRGB, cv2.COLOR_BGR2HSV) # what the fuck
imBW = cv2.cvtColor(imRGB, cv2.COLOR_BGR2GRAY) 

os.chdir(positiveOutputPath)
posMask = cv2.inRange(imHSV, lower_red_HSV, upper_red_HSV)
posRes = cv2.bitwise_and(imBGR, imBGR, mask = posMask)
cv2.imwrite(file, posRes)

plot.pyplot.imshow(imRGB)
plot.pyplot.show()

plot.pyplot.imshow(posRes)
plot.pyplot.show()

os.chdir(negativeOutputPath)
negMask = cv2.bitwise_not(posMask)
negRes = cv2.bitwise_and(imBGR, imBGR, mask = negMask)
cv2.imwrite(file, negRes)