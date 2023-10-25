import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

original_image = Image.open('./Media/gray3.jpg')
image = np.array(original_image)

def fourier_transform(image):
    # Discrete Fourier Transform
    dft = np.fft.fft2(image)

    # Shifting the zero frequency component to center
    dft_shifted = np.fft.fftshift(dft)

    # Computing the Magnitude Spectrum
    magnitude_spectrum = np.abs(dft_shifted)

    return magnitude_spectrum

# Display the magnitude spectrum
magnitude_spectrum = fourier_transform(image)

output_fourier_img = Image.fromarray(np.uint8(255 * (np.log(magnitude_spectrum + 1) / np.max(np.log(magnitude_spectrum + 1)))))
output_fourier_img.save('./Media/output_fourier.jpg')

def inverse_fourier_transform(image):
    # Shift the zero frequency component back to the top-left
    dft_shifted = np.fft.ifftshift(image)

    # Take the inverse Fourier Transform
    output_image = np.fft.ifft2(dft_shifted)

    # Convert the complex values to magnitude (absolute values)
    output_image = np.abs(output_image)

    # Normalize the image
    output_image = (output_image - np.min(output_image)) / (np.max(output_image) - np.min(output_image)) * 255

    # Convert to 8-bit unsigned integer
    output_image = np.uint8(output_image)

    output_image = Image.fromarray(output_image)
    return output_image

output_inverse_fourier_img = inverse_fourier_transform(image)
output_inverse_fourier_img.save('./Media/output_inverse_fourier.jpg')