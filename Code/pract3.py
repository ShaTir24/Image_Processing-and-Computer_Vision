import numpy as np
from PIL import Image, ImageFilter

input_mean_img = Image.open('./Media/gray.jpg')
input_mean_img = np.array(input_mean_img)

input_median_img = Image.open('./Media/noisy_img.jpg')
input_median_img = np.array(input_median_img)

kernel_size = 3

# Function to implement Mean Filter
def mean_filter(image, filter_size):
    kernel = np.ones((filter_size, filter_size))
    kernel /= np.sum(kernel)

    height, width = image.shape

    output_image = np.zeros((height, width), dtype=np.uint8)

    # Apply the mean filter to the image
    for i in range(height - filter_size + 1):
        for j in range(width - filter_size + 1):
            # Region of Kernel (Applying mask)
            rok = image[i:i+filter_size, j:j+filter_size]

            # Apply the filter by element-wise multiplication and summation
            output_pixel = np.sum(rok * kernel)

            # Place the result in the output image
            output_image[i + filter_size // 2, j + filter_size // 2] = int(output_pixel)

    output_image = Image.fromarray(output_image)
    return output_image

output_mean_img = mean_filter(input_mean_img, kernel_size)
output_mean_img.save('./Media/output_mean.jpg')


# Function to implement Median Filter
def median_filter(image, filter_size):
    height, width = image.shape

    # Create an output array with the same shape as the input
    output_array = np.zeros_like(image)

    # Apply the median filter
    for i in range(height):
        for j in range(width):
            kernel = image[max(0, i - filter_size // 2):min(height, i + filter_size // 2 + 1),
                             max(0, j - filter_size // 2):min(width, j + filter_size // 2 + 1)]
        
            output_array[i, j] = np.median(kernel)

    # Create a Pillow image from the filtered array
    output_image = Image.fromarray(output_array)
    return output_image

output_median_img = median_filter(input_median_img, kernel_size)
output_median_img.save('./Media/output_median.jpg')

def gaussian_blur(image_path):
    image = Image.open(image_path)
    output_img = image.filter(ImageFilter.GaussianBlur(radius = 3))
    return output_img

output_gaussian_blur_image = gaussian_blur('./Media/sample3.jpg')
output_gaussian_blur_image.save('./Media/output_gaussian.jpg')

# Function to implement Laplacian Filter
def laplacian_filter(image):
    laplacian_kernel = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]])
    height, width = image.shape
    pad = len(laplacian_kernel) // 2 # Padding size for borders
    # Create an empty result image
    output_image = np.zeros((height, width), dtype=np.uint8)

    # Convolve the image
    for y in range(pad, height - pad):
        for x in range(pad, width - pad):
            output_image[y, x] = np.sum(image[y - pad:y + pad + 1, x - pad:x + pad + 1] * laplacian_kernel)
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    output_image = Image.fromarray(output_image)
    return output_image

output_laplace_img = laplacian_filter(input_mean_img)
output_laplace_img.save('./Media/output_laplace.jpg')