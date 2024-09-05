
This repository focuses on Hough Transform and Homography Estimation.

# Project Overview
### `Hough Transform Implementation:`

Objective: Detect straight lanes in images using the Hough Transform method.

Files:

### hough.py:
Contains the script for detecting lanes using Hough Transform.

### road.jpg: 
Input image used for lane detection.

### utils.py: 
Utility functions including create_line and create_mask.

### `Hough Transform details:`

#### Edge Detection: The hough.py script uses the Canny edge detector to find edges in the input image.

##### Hough Transform: Computes the Hough space to detect lane lines.

#### Lane Detection: Identifies two major lane lines and applies non-max suppression to enhance accuracy.

#### Results: Plots of the original image, edges, ROI edges, and detected lanes.
#### Images
![image](https://github.com/user-attachments/assets/152d44f2-dda5-4d9b-9d43-2ab09d87b0dc)
![image](https://github.com/user-attachments/assets/b96c3ed0-441c-42ef-a5f1-5fe4709318d5)
![image](https://github.com/user-attachments/assets/eafadeed-f108-4469-81c7-ac7620ea8fbe)


### `Homography Estimation:`

### Objective: 
Estimate the homography between images and use it for an augmented reality application.

Files:
### homography.py:
Contains functions for matching keypoints, computing homography with RANSAC, and warping images.
### run.py:
Script to execute the homography estimation and image warping.
### cv_cover.jpg, cv_desk.jpg, hp_cover.jpg: 
Images used for homography estimation and compositing.


### `Homography Estimation Details:`

#### Keypoint Matching: Uses SIFT to find matching keypoints between images.

#### RANSAC: Implements RANSAC to estimate the homography matrix from the keypoint matches.

#### Image Warping: Applies the estimated homography to warp images and composite them for an augmented reality effect.

#### Results: Visualization of the final composited image.
#### Images:
![image](https://github.com/user-attachments/assets/e2beb9d5-60ce-4699-94ea-6543fc053b19)
![image](https://github.com/user-attachments/assets/a71a6988-91fb-4fa2-a039-a0efa343876b)
![image](https://github.com/user-attachments/assets/03885910-0be5-4268-8f95-c9ca0b7d1ad1)
![image](https://github.com/user-attachments/assets/9b9a182e-28c2-4032-ba93-533412ea6d80)



# Dependencies
numpy,
opencv-python,
matplotlib,
scikit-image,

Install the required packages using:


-pip install numpy opencv-python matplotlib scikit-image

# License
This project is licensed under the MIT License.
