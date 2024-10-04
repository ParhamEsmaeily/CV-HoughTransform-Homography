# import other necessary libaries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import canny
from utils import create_line, create_mask
# load the input image
image = cv2.imread('road.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# run Canny edge detector to find edge points
edges = canny(gray)
# create a mask for ROI by calling create_mask
mask = create_mask(edges.shape[0], edges.shape[1])
# extract edge points in ROI by multipling edge map with the mask
edges_roi = edges * mask
# perform Hough transform
#lines = cv2.HoughLines(edges_roi.astype(np.uint8), rho=1, theta=np.pi / 180, threshold=95)
# Perform Hough transform
theta = np.deg2rad(np.arange(-90.0, 90.0))
width, height = edges_roi.shape
diag_len = int(np.ceil(np.sqrt(width * width + height * height)))
rho = np.linspace(-diag_len, diag_len, int(diag_len * 2.0))
H = np.zeros((len(rho), len(theta)))
for y in range(width):
    for x in range(height):
        if edges_roi[y, x]:
            for theta_index, angle in enumerate(theta):
                rho_val = x * np.cos(angle) + y * np.sin(angle)
                rho_index = np.argmin(np.abs(rho - rho_val))
                H[rho_index, theta_index] += 1
# find the right lane by finding the peak in hough space
max_index = np.argmax(H)
rho_index, theta_index = np.unravel_index(max_index, H.shape)
rho_val = rho[rho_index]
theta_val = theta[theta_index]
xs1, ys1 = create_line(rho_val, theta_val, edges_roi)

# zero out the values in accumulator around the neighborhood of the peak
neighborhood_size = 100
H[max(0, rho_index - neighborhood_size):min(H.shape[0], rho_index + neighborhood_size),
  max(0, theta_index - neighborhood_size):min(H.shape[1], theta_index + neighborhood_size)] = 0
# find the left lane by finding the peak in hough space
max_index = np.argmax(H)
rho_index, theta_index = np.unravel_index(max_index, H.shape)
rho_val = rho[rho_index]
theta_val = theta[theta_index]
xs, ys = create_line(rho_val, theta_val, edges_roi)

# plot the results


#Perform Hough transform
# Find the most dominant lane line (the one with the highest count)

# Find the two most dominant lane lines
#if lines is not None:
    # Convert lines to (rho, theta) format
    #lines = np.squeeze(lines, axis=1)
    # Group lines by angle
    #grouped_lines = {}
    #for rho, theta in lines:
        #angle = np.degrees(theta)
        #if angle not in grouped_lines:
           # grouped_lines[angle] = []
       # grouped_lines[angle].append(rho)

    # Find the angle with the most lines
    #dominant_angle = max(grouped_lines, key=lambda angle: len(grouped_lines[angle]))

    # Find the median rho value for the dominant angle
    #median_rho = np.median(grouped_lines[dominant_angle])

    # Draw the detected line corresponding to the dominant angle and median rho
    #xs, ys = create_line(median_rho, np.radians(dominant_angle), image)
    #if xs and ys:
        #cv2.line(image, (xs[0], ys[0]), (xs[-1], ys[-1]), (150, 100, 0), 5)

# Display the result



# Plot the results
plt.figure(figsize=(12, 8))

# Original Image
#plt.subplot(2, 2, 1)
#plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#plt.title('OG Image')
#plt.axis('off')

# Canny Edges
plt.subplot()
plt.imshow(edges, cmap='gray')
plt.title('Edges')
plt.axis('off')
plt.show()

plt.subplot()
plt.imshow(mask, cmap='gray')
plt.title('Mask')
plt.axis('off')
plt.show()

# ROI Edges
plt.subplot()
plt.imshow(edges_roi, cmap='gray')
plt.title('Edges in ROI')
plt.axis('off')
plt.show()

# Lanes Detected
plt.subplot()

#plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#xs, ys = create_line(rho, theta, image)
#plt.plot(xs, ys, color='orange', linewidth=2.5)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.plot(xs1, ys1, color='blue', linewidth=2)
plt.plot(xs, ys, color='orange', linewidth=2)
plt.title('Detected Lanes')
plt.axis('off')
plt.show()
