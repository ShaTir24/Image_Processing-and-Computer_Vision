from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Histogram Stretching

# Method to process the red band of the image
def normalizeRed(intensity):
    iI      = intensity
    minI    = 86
    maxI    = 230
    minO    = 0
    maxO    = 255
    iO      = (iI-minI)*(((maxO-minO)/(maxI-minI))+minO)
    return iO

# Method to process the green band of the image
def normalizeGreen(intensity):
    iI      = intensity
    minI    = 90
    maxI    = 225
    minO    = 0
    maxO    = 255
    iO      = (iI-minI)*(((maxO-minO)/(maxI-minI))+minO)
    return iO

# Method to process the blue band of the image
def normalizeBlue(intensity):
    iI      = intensity
    minI    = 100
    maxI    = 210
    minO    = 0
    maxO    = 255
    iO      = (iI-minI)*(((maxO-minO)/(maxI-minI))+minO)
    return iO

imageObject     = Image.open("./Media/cropped_image.jpg")

# Split the red, green and blue bands from the Image
multiBands      = imageObject.split()

# Apply point operations that does contrast stretching on each color band
normalizedRedBand      = multiBands[0].point(normalizeRed)
normalizedGreenBand    = multiBands[1].point(normalizeGreen)
normalizedBlueBand     = multiBands[2].point(normalizeBlue)

# Create a new image from the contrast stretched red, green and blue brands
normalizedImage = Image.merge("RGB", (normalizedRedBand, normalizedGreenBand, normalizedBlueBand))

# Display and Saving the image after contrast stretching
normalizedImage.show()
normalizedImage.save('./Media/equalized.jpg')


# Histogram Equalization

def histogram_equalization(image):
    img_array = np.array(image)
    # Compute the histogram of the image
    histogram, _ = np.histogram(img_array, bins=256, range=(0, 255))

    # Compute the cumulative distribution function (CDF) of the histogram
    cdf = histogram.cumsum()

    # Normalize the CDF to a 0-255 range
    cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())

    # Interpolate the CDF to get the equalization mapping
    img_equalized = cdf[img_array]

    # Create a new PIL image from the equalized NumPy array
    equalized_image = Image.fromarray(np.uint8(img_equalized))

    # Save the equalized image
    equalized_image.save('./Media/output_equalized.jpg')

    # Plot the histograms
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title('Original Image Histogram')
    plt.hist(img_array.flatten(), bins=256, range=(0, 255), density=True, color='b', alpha=0.6)
    plt.subplot(122)
    plt.title('Equalized Image Histogram')
    plt.hist(img_equalized.flatten(), bins=256, range=(0, 255), density=True, color='r', alpha=0.6)
    plt.tight_layout()
    plt.show()

image = Image.open('./Media/gray.jpg')
histogram_equalization(image)