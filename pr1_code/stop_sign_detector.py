'''
ECE276A WI20 HW1
Stop Sign Detector
'''

import os, cv2
import numpy as np
import skimage

class StopSignDetector():
    def __init__(self):
        '''
            Initilize your stop sign detector with the attributes you need,
            e.g., parameters of your classifier
        '''
        self.weights = np.array([-1198.70681915, 2975.83862542, -19176.06813495, 9368.96722851]) # removed a third of the red pixels so the entire dataset would be evenly split 50/50
        
        self.ARC_COEFF = 0.01
        self.MIN_PTS_IN_CONTOUR = 8
        self.MAX_PTS_IN_CONTOUR = 12
        self.MIN_AREA_RATIO = 0.00015
        self.MAX_ECCENTRICITY = 0.61
        self.MAX_SIGNS_PER_IMG = 2
        self.KERNEL_SIZE = 3

    def segment_image(self, img):
        '''
            Obtain a segmented image using a color classifier,
            Logistic Regression

            Inputs:
                img - original image
            Outputs:
                mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
        '''

        x = img.flatten().reshape(img.shape[0] * img.shape[1], img.shape[2])
        
        intercept = np.ones((x.shape[0], 1))
        x = np.concatenate((intercept, x), axis = 1) # add bias column
        
        mask_img = np.matmul(x, self.weights)
        mask_img = mask_img.reshape(img.shape[0], img.shape[1])
        mask_img = 1.0 * (mask_img > 0) # binarize image
        mask_img = skimage.img_as_ubyte(mask_img) # convert to uint8

        return mask_img

    def get_bounding_box(self, img):
        '''
            Find the bounding box of the stop sign
            call other functions in this class if needed

            Inputs:
                img - original image
            Outputs:
                boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2]
                where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
                is from left to right in the image.

            Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
        '''
        boxes = []
        mask_img = self.segment_image(img)
        img_area = img.shape[0] * img.shape[1]
        
        # get thresholded image and blur. Canny, Sobel, Laplacian all had worse performance
        _, thresh_img = cv2.threshold(mask_img, 0, 255, cv2.THRESH_BINARY)
        
        # Gaussian blur and erosion for connected, smooth contours
        kernel = (self.KERNEL_SIZE, self.KERNEL_SIZE)
        smoothed_img = cv2.GaussianBlur(thresh_img, kernel, cv2.BORDER_DEFAULT)
        eroded_img = cv2.erode(smoothed_img, kernel, iterations = 1)
        
        # find largest contours
        contours, _ = cv2.findContours(eroded_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        contours = sorted(contours, key = cv2.contourArea, reverse = True)[:self.MAX_SIGNS_PER_IMG]
        
        for contour in contours:
            perimeter = self.ARC_COEFF * cv2.arcLength(contour, True)
            poly_approx = cv2.approxPolyDP(contour, perimeter, True)
            
            if (len(poly_approx) >= self.MIN_PTS_IN_CONTOUR):
                _, axes, _ = cv2.fitEllipse(poly_approx)
                
                eccentricity = np.sqrt(1 - np.square(min(axes) / max(axes)))
                
                if (len(poly_approx) <= self.MAX_PTS_IN_CONTOUR or eccentricity <= self.MAX_ECCENTRICITY):
                    (x, y, w, h) = cv2.boundingRect(poly_approx)
                    area = w * h
                    
                    if ((area / img_area) >= self.MIN_AREA_RATIO):
                        bounds = [x, img.shape[0] - y - h, x + w, img.shape[0] - y] # convert coordinates to project specification
                        boxes.append(bounds)
            
#            out = cv2.drawContours(img, [contour], -1, (0, 255, 0), 3)
        print(boxes)
        return boxes

if __name__ == '__main__':
    folder = "trainset"
    my_detector = StopSignDetector()
    for filename in os.listdir(folder):
        # read one test image
        img = cv2.imread(os.path.join(folder, filename))
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Display results:
    # (1) Segmented images
    #	 mask_img = my_detector.segment_image(img)
    # (2) Stop sign bounding box
    #    boxes = my_detector.get_bounding_box(img)
    # The autograder checks your answers to the functions segment_image() and get_bounding_box()
    # Make sure your code runs as expected on the testset before submitting to Gradescope

