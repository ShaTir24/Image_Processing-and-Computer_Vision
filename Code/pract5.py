import cv2
import numpy as np
import matplotlib.pyplot as plt

# For HOG Operations
from skimage.io import imread
from skimage.feature import hog

# Implementing SIFT Features

image1 = cv2.imread('./Media/books.jpg')
image2 = cv2.imread('./Media/book1.jpg')

# Function for Keypoints Detection in an image
def detect_keypoints(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)

    # Displaying the Keypoints in the image
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_temp = cv2.drawKeypoints(gray_img, keypoints, image)
    cv2.imshow("Image", img_temp)
    cv2.waitKey(0)
    return keypoints, descriptors

# Function to match keypoint detectors in two images
def match_features(descriptors1, descriptors2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.65 * n.distance:
            good_matches.append(m)

    return good_matches

# Function to display the feature matching projection
def visualize_matches(image1, image2, keypoints1, keypoints2, matches):
    result = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None)
    cv2.imshow("Matches", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

keypoints1, descriptors1 = detect_keypoints(image1)
keypoints2, descriptors2 = detect_keypoints(image2)

matches = match_features(descriptors1, descriptors2)

visualize_matches(image1, image2, keypoints1, keypoints2, matches)

# Implementing HOG Features

img = imread("./Media/gray.jpg")

# Function for applying the hog algorithm
def hog_features(img):
    fd, hog_image = hog(img,
                    orientations=16,
                    pixels_per_cell=(4, 4),
                    cells_per_block=(3,3),
                    visualize=True)

    plt.axis("off")
    plt.imshow(hog_image, cmap='gray')
    plt.savefig("./Media/self_hog.png")
    plt.show()

hog_features(img)