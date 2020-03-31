This project is composed of the following files

- stop_sign_detector.py
- stop_sign_regression.py
- stop_sign_red_filter_imgs.py
- stop_sign_red_dataset_formatter.py
- stop_sign_notRed_dataset_formatter.py

stop_sign_detector.py
	- Contains 2 functions
		- segment_image(img) which takes and image and returns a binarized image with only red pixels colored white
		- get_bounding_box(img) which takes an image, calls segment_img(img), and returns a list of rectangular boxes in the specified format.
	- Contains several parameter values that can be changed to achieve different results:
		- ARC_COEFF = the coefficient of the arclength for the approxPolyDP functions
		- MIN_PTS_IN_CONTOUR = the minimum number of points I consider necessary to be possibly considered a stop-sign
		- MAX_PTS_IN_CONTOUR = the maximum number of points I consider necessary to be possibly considered a stop-sign
		- MIN_AREA_RATIO = the proportion of the area of the red segment of the image to the area of the total image
		- MAX_ECCENTRICITY = the greatest permissible eccentricity of the approximated ellipsoid to be possibly considered a stop-sign
		- MAX_SIGNS_PER_IMG = the maximum possible number of stop-signs that may occur in an image
		- KERNEL_SIZE = the size of the kernel for the Gaussian Blur and erosion
		
stop_sign_regression.py
	- Script that takes the dataset formed from the dataset_formatter.py files and performs the logistic regression.
	- Also logs the loss over iterations in a variable for easy use later (losses.npy).
	- Use this to generate weights to be used during classification (omega.npy).
	
stop_sign_red_filter_imgs.py
	- Script that converts all images in a directory to HSV, filters according to preset HSV value bounds, and exports the segmented images. The dataset_formatter.py files take the images that are output from this file to form the datasets that stop_sign_regression.py uses.
	
dataset_formatter.py
	- Simply takes in an image directory and outputs an npy file for the stop_sign_regression.py file to take in.
