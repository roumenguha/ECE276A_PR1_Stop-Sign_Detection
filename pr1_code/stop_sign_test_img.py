# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 11:34:48 2020

@author: roume
"""

import os, cv2
import numpy as np
from matplotlib import pyplot as plt
import skimage

inputPath = 'C:/Users/roume/PycharmProjects/ECE276A_Project1_StopSignDetector/venv/dataset'
inputFile = "t5.jpg"

masksOutputPath = 'C:/Users/roume/PycharmProjects/ECE276A_Project1_StopSignDetector/venv/Results/masks'
boundedOutputPath = 'C:/Users/roume/PycharmProjects/ECE276A_Project1_StopSignDetector/venv/Results/bounded'

weights = np.array([-1198.70681915, 2975.83862542, -19176.06813495, 9368.96722851]) # removed a third of the red pixels so the entire dataset would be evenly split 50/50

ARC_COEFF = 0.01
MIN_PTS_IN_CONTOUR = 8
MAX_PTS_IN_CONTOUR = 12
MIN_AREA_RATIO = 0.00015
MAX_ECCENTRICITY = 0.61
MAX_SIGNS_PER_IMG = 2
KERNEL_SIZE = 3

###############################################################################
boxes = []

os.chdir(inputPath)
img = cv2.imread(inputFile)
img_area = img.shape[0] * img.shape[1]

x = img.flatten().reshape(img.shape[0] * img.shape[1], img.shape[2])

# Add bias column
intercept = np.ones((x.shape[0], 1))
x = np.concatenate((intercept, x), axis = 1)

# generate binary mask
mask_img = np.matmul(x, weights)
mask_img = mask_img.reshape(img.shape[0], img.shape[1])
mask_img = 1.0 * (mask_img > 0) 
mask_img = skimage.img_as_ubyte(mask_img)

# get thresholded image. Canny had worse performance
ret, thresh_img = cv2.threshold(mask_img, 0, 255, cv2.THRESH_BINARY)

# Gaussian blur for smooth contours
smoothed_img = cv2.GaussianBlur(thresh_img, (3, 3), cv2.BORDER_DEFAULT)

# find contours
contours, heirarchy = cv2.findContours(smoothed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

#laplacian = cv2.Laplacian(smoothed_img, cv2.CV_64F)
#sobelx = cv2.Sobel(smoothed_img, cv2.CV_64F, 1, 0, ksize = 5)
#sobely = cv2.Sobel(smoothed_img, cv2.CV_64F, 0, 1, ksize = 5)

#plt.subplot(2,2,1),plt.imshow(smoothed_img, cmap = 'gray')
#plt.title('Original'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,2,2),plt.imshow(laplacian, cmap = 'gray')
#plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,2,3),plt.imshow(sobelx, cmap = 'gray')
#plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,2,4),plt.imshow(sobely, cmap = 'gray')
#plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

#plt.show()

## Output dtype = cv2.CV_8U
#sobelx8u = cv2.Sobel(smoothed_img, cv2.CV_8U, 1, 0, ksize = 3)
#
## Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
#sobelx64f = cv2.Sobel(smoothed_img, cv2.CV_64F, 1, 0, ksize = 3)
#
#abs_sobel64f = np.absolute(sobelx64f)
#sobel_8u = np.uint8(abs_sobel64f)
#
#plt.subplot(1,3,1),plt.imshow(smoothed_img, cmap = 'gray')
#plt.title('Original'), plt.xticks([]), plt.yticks([])
#plt.subplot(1,3,2),plt.imshow(sobelx8u, cmap = 'gray')
#plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
#plt.subplot(1,3,3),plt.imshow(sobel_8u, cmap = 'gray')
#plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])
#
#plt.show()

contours = sorted(contours, key = cv2.contourArea, reverse = True)[:MAX_SIGNS_PER_IMG]
contoured_img = np.zeros(img.shape)

for contour in contours:        
    img_area = img.shape[0] * img.shape[1]
    
    # get thresholded image and blur. Canny, Sobel, Laplacian all had worse performance
    _, thresh_img = cv2.threshold(mask_img, 0, 255, cv2.THRESH_BINARY)
    
    # Gaussian blur and erosion for connected, smooth contours
    kernel = (KERNEL_SIZE, KERNEL_SIZE)
    smoothed_img = cv2.GaussianBlur(thresh_img, kernel, cv2.BORDER_DEFAULT)
    eroded_img = cv2.erode(smoothed_img, kernel, iterations = 1)
    
    # find largest contours
    contours, _ = cv2.findContours(eroded_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:MAX_SIGNS_PER_IMG]
    
    for contour in contours:
        perimeter = ARC_COEFF * cv2.arcLength(contour, True)
        poly_approx = cv2.approxPolyDP(contour, perimeter, True)
        
        if (len(poly_approx) >= MIN_PTS_IN_CONTOUR):
            _, axes, _ = cv2.fitEllipse(poly_approx)
            
            eccentricity = np.sqrt(1 - np.square(min(axes) / max(axes)))
            
            if (len(poly_approx) <= MAX_PTS_IN_CONTOUR or eccentricity <= MAX_ECCENTRICITY):
                (x, y, w, h) = cv2.boundingRect(poly_approx)
                area = w * h
                
                if ((area / img_area) >= MIN_AREA_RATIO):
                    bounds = [x, img.shape[0] - y - h, x + w, img.shape[0] - y] # convert coordinates to project specification
                    boxes.append(bounds)
        
        out = cv2.drawContours(img, [contour], -1, (0, 255, 0), 3)
    
plt.imshow(out)

os.chdir(masksOutputPath)
cv2.imwrite(inputFile, mask_img)

os.chdir(boundedOutputPath)
cv2.imwrite(inputFile, out)

