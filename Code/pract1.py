from PIL import Image
import numpy as np

im = Image.open(r"T:\\Local Data\\L.D.C.E. Sem7\\placed.jpg")
width, height = im.size
px = im.load()

# resizing image to a shorter dimension
size = (200, 250)
im1 = im.resize(size)
px1 = im1.load()

# Function to read the entire image and printing all the pixel values
def readImg():
    for i in range (height):
        for j in range (width):
            #pixel = im.getpixel((j, i))
            pixel = px[j, i]
            #print(f"Pixel at {j}, {i} is {pixel}")
            print(f"Pixel at {j}, {i} = {pixel}")
        print("\n")

# Function to crop the image and print the pixels and writing the image to local disk
def cropImg():
    # Cropping the image in 200x250 resolution
    for i in range (200):
        for j in range (250):
            pixel = px1[i, j]
            print(f"Pixel at {j}, {i} = {pixel}")
        print("\n")
    
    # writing the image to local storage
    im1.save('.Media/cropped_image.jpg')

# Function to convert image into grayscale
def grayImg():
    grayimg = np.empty([250, 200], dtype=np.uint8)
    for i in range (200):
        for j in range (250):
            pixel = px1[i, j]
            grayimg[j][i] = int((pixel[0] + pixel[1] + pixel[2]) / 3)
            #grayimg[j][i] = int(pixel[0]*0.2126 + pixel[1]*0.7152 + pixel[2]*0.0722)

    im_gray = Image.fromarray(grayimg)
    im_gray.save('./Media/gray.jpg')

# function to convert coloured image to binary (black and white)
def bwImg():
    binary_img = np.empty([250, 200], dtype=np.bool_)
    for i in range (200):
        for j in range (250):
            pixel = px1[i,j]
            avg = int((pixel[0] + pixel[1] + pixel[2]) / 3)
            if (avg < 128):
                binary_img[j][i] = 0
            else:
                binary_img[j][i] = 255

    im_binary = Image.fromarray(binary_img)
    im_binary.save('./Media/b&w.jpg')

# function to complement the image colours
def compImg():
    comp_img = np.empty([250, 200, 3], dtype=np.uint8)
    for i in range (200):
        for j in range (250):
            pixel = px1[i, j]
            comp_img[j][i][0] = 255 - pixel[0]
            comp_img[j][i][1] = 255 - pixel[1]
            comp_img[j][i][2] = 255 - pixel[2]

    im_comp = Image.fromarray(comp_img)
    im_comp.save('./Media/Complement.jpg')