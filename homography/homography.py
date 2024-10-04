import matplotlib.pyplot as plt
import numpy as np
from skimage import data
from skimage import transform
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, plot_matches, SIFT
from skimage.transform import warp
from skimage.morphology import binary_dilation, disk
def matchPics(I1, I2):
    # Given two images I1 and I2, perform SIFT matching to find candidate match pairs

    ### YOUR CODE HERE
    ### You can use skimage or OpenCV to perform SIFT matching
    #I1 = transform.rotate(I1, 90)
    I2 = rgb2gray(I2)

    descriptor_extractor = SIFT()

    descriptor_extractor.detect_and_extract(I1)
    locs1 = descriptor_extractor.keypoints
    descriptors1 = descriptor_extractor.descriptors

    descriptor_extractor.detect_and_extract(I2)
    locs2 = descriptor_extractor.keypoints
    descriptors2 = descriptor_extractor.descriptors



    matches = match_descriptors(descriptors1, descriptors2, max_ratio=0.6,
                                  cross_check=True)



    ### END YOUR CODE
    print(("Locs1 ", locs1, "| Locs2: ", locs2, "| matches: ", matches, " end\n"))
    return matches, locs1, locs2

def computeH_ransac(matches, locs1, locs2):

    # Compute the best fitting homography using RANSAC given a list of matching pairs

    ### YOUR CODE HERE
    ### You should implement this function using Numpy only
    bestH = None
    best_inliers = np.array([])
    num_iterations = 1000
    inlier_threshold = 5

    for _ in range(num_iterations):
        #reverse because of sift skimmage took me a day to fix, this will make bound box correct.
        locs1[:, [0, 1]] = locs1[:, [1, 0]]
        locs2[:, [0, 1]] = locs2[:, [1, 0]]
        # Randomly select minimal subset of matches
        random_indices = np.random.choice(len(matches), 70, replace=False)



        src_points = locs1[matches[random_indices, 0], :2]

        dst_points = locs2[matches[random_indices, 1], :2]

        # Compute homography matrix
        A = np.zeros((8, 9))
        for i in range(4):
            A[2 * i, :] = [-src_points[i][0], -src_points[i][1], -1, 0, 0, 0,
                           src_points[i][0] * dst_points[i][0], src_points[i][1] * dst_points[i][0], dst_points[i][0]]
            A[2 * i + 1, :] = [0, 0, 0, -src_points[i][0], -src_points[i][1], -1,
                               src_points[i][0] * dst_points[i][1], src_points[i][1] * dst_points[i][1],
                               dst_points[i][1]]

        _, _, V = np.linalg.svd(A)
        H = V[-1, :].reshape(3, 3)


        # Compute inliers
        src_points_homogeneous = np.concatenate([src_points, np.ones((70, 1))], axis=1)
        transformed_points = np.dot(H, src_points_homogeneous.T).T
        transformed_points /= transformed_points[:, 2][:, None]
        distances = np.linalg.norm(transformed_points[:, :2] - dst_points, axis=1)
        inliers = np.where(distances < inlier_threshold)[0]

        #print("len(inliers):  ",len(inliers))


        #print("len(best_inliers):  ",len(best_inliers))
        #print()

        if len(inliers) > len(best_inliers):

            bestH = H
            print("3x3 ",bestH)

            best_inliers = inliers
    ### END YOUR CODE

    ##Printing best 3x3
    print("BestH is: ", bestH)
    return bestH, inliers



def compositeH(H, template, img):
    # Create a composite image after warping the template image on top
    # of the image using homography

    # Create mask of same size as template
    mask = np.ones_like(template)

    # Warp mask by appropriate homography
    warped_mask = warp(mask, np.linalg.inv(H), output_shape=img.shape[:2])

    # Warp template by appropriate homography
    warped_template = warp(template, np.linalg.inv(H), output_shape=img.shape)

    # Use mask to combine the warped template and the image
    composite_img = img * (1 - warped_mask) + warped_template
    return composite_img
