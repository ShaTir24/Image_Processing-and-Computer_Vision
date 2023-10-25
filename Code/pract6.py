import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image = Image.open('./Media/gray.jpg')
img_array = np.array(image)

# Function to implement Sobel's Edge Segmentation
def sobel_edge_detection(img_array):
    # Sobel operators for horizontal and vertical edges
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

     # Get the size of the image
    height, width = img_array.shape

    # Create arrays to store the results
    edges_x = np.zeros((height, width))
    edges_y = np.zeros((height, width))

    # Perform the convolution
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            patch = img_array[i-1:i+2, j-1:j+2]
            edges_x[i, j] = np.sum(patch * sobel_x)
            edges_y[i, j] = np.sum(patch * sobel_y)

    # Combine horizontal and vertical edges
    edge_image = np.sqrt(edges_x**2 + edges_y**2)

    # Normalize the edge image
    edge_image = (edge_image / edge_image.max()) * 255

    edge_image = Image.fromarray(np.uint8(edge_image))

    plt.imshow(edge_image, cmap='gray')
    plt.axis('off')
    plt.show()

    edge_image.save('./Media/output_edge_based.jpg')

sobel_edge_detection(img_array)

# Function to implement Otsu's Threshold (Frequency based) segmentation
def otsu_thresholding(img_array):

    # Calculate the histogram of the grayscale image
    histogram, bins = np.histogram(img_array, bins=256, range=(0, 256))

    # Calculate the probabilities of each intensity level
    pixel_count = img_array.shape[0] * img_array.shape[1]
    probabilities = histogram / pixel_count

    # Calculate the cumulative probabilities
    cumulative_probabilities = np.cumsum(probabilities)

    # Calculate the mean intensity
    intensity_values = np.arange(256)
    mean_intensity = np.sum(intensity_values * probabilities)

    # Initialize variables to store the best threshold and maximum between-class variance
    best_threshold = 0
    max_variance = 0

    for threshold in range(256):
        # Calculate the probabilities for the two classes (background and foreground)
        prob_background = cumulative_probabilities[threshold]
        prob_foreground = 1 - prob_background

        # Calculate the means for the two classes
        mean_background = np.sum(intensity_values[:threshold + 1] * probabilities[:threshold + 1]) / prob_background
        mean_foreground = (mean_intensity - mean_background * prob_background) / prob_foreground

        # Calculate the between-class variance
        between_class_variance = prob_background * prob_foreground * (mean_background - mean_foreground) ** 2

        # Check if the between-class variance is greater than the maximum found so far
        if between_class_variance > max_variance:
            max_variance = between_class_variance
            best_threshold = threshold

    # Apply the threshold to the image
    thresholded_image = img_array > best_threshold

    thresholded_image = Image.fromarray((thresholded_image * 255).astype(np.uint8))

    plt.imshow(thresholded_image, cmap='gray')
    plt.axis('off')
    plt.show()
    thresholded_image.save('./Media/output_thresholded.jpg')

otsu_thresholding(img_array)

# Function to implement Region Based segmentation
def region_growing_segmentation(img_array, seed_pixel, threshold):

    # Initialize an empty segmentation mask
    segmentation_mask = np.zeros_like(img_array, dtype=bool)

    # Get image dimensions
    height, width = img_array.shape

    # Define a queue for region growing
    queue = []

    # Define the 8-connectivity neighbors
    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    # Set the seed pixel as the starting point
    queue.append(seed_pixel)

    while queue:
        x, y = queue.pop(0)

        if not segmentation_mask[x, y]:
            # Check the intensity difference between the current pixel and the seed pixel
            if abs(int(img_array[x, y]) - int(img_array[seed_pixel])) <= threshold:
                segmentation_mask[x, y] = True

                # Add 8-connectivity neighbors to the queue
                for dx, dy in neighbors:
                    new_x, new_y = x + dx, y + dy

                    if 0 <= new_x < height and 0 <= new_y < width:
                        queue.append((new_x, new_y))

    plt.imshow(segmentation_mask, cmap='gray')
    plt.axis('off')
    plt.show()
    segmented_image = Image.fromarray(segmentation_mask)
    segmented_image.save('./Media/output_regional.jpg')

region_growing_segmentation(img_array, (100, 120), 30)    #seed-pixel, threshold