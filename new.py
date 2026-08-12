import cv2

img = cv2.imread("image.jpg")
#2.3

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

cv2.imshow("Gray", gray)
cv2.imshow("Black & White", bw)

cv2.waitKey(0)
cv2.destroyAllWindows()

#2.4

#flipCode = 0 → Vertical flip (top ↔ bottom)
#flipCode = 1 → Horizontal flip (left ↔ right)
#flipCode = -1 → Both horizontal and vertical


# Horizontal flip
horizontal = cv2.flip(img, 1)

# Vertical flip
vertical = cv2.flip(img, 0)

# Display images
cv2.imshow("Original", img)
cv2.imshow("Horizontal Flip", horizontal)
cv2.imshow("Vertical Flip", vertical)

cv2.waitKey(0)
cv2.destroyAllWindows()

#2.5

import cv2

# Read image
img = cv2.imread("image.jpg")

# Crop different portions
crop1 = img[0:200, 0:200]         # Top-left
crop2 = img[100:300, 150:350]     # Center
crop3 = img[250:450, 300:500]     # Bottom-right

# Display
cv2.imshow("Original", img)
cv2.imshow("Crop 1", crop1)
cv2.imshow("Crop 2", crop2)
cv2.imshow("Crop 3", crop3)

# Save cropped images
cv2.imwrite("crop1.jpg", crop1)
cv2.imwrite("crop2.jpg", crop2)
cv2.imwrite("crop3.jpg", crop3)

cv2.waitKey(0)
cv2.destroyAllWindows()


from PIL import Image

# Read image
img = Image.open("image.jpg")

# Crop different portions
crop1 = img.crop((0, 0, 200, 200))
crop2 = img.crop((150, 100, 350, 300))
crop3 = img.crop((300, 250, 500, 450))

# Display
img.show()
crop1.show()
crop2.show()
crop3.show()

# Save
crop1.save("crop1.jpg")
crop2.save("crop2.jpg")
crop3.save("crop3.jpg")

#2.6

#M1

import cv2

# Read image
img = cv2.imread("image.jpg")

# Find negative
negative = 255 - img

# Display
cv2.imshow("Original", img)
cv2.imshow("Negative", negative)

# Save
cv2.imwrite("negative.jpg", negative)

cv2.waitKey(0)
cv2.destroyAllWindows()

#M2
import cv2

# Read image
img = cv2.imread("image.jpg")

# Find negative
negative = cv2.bitwise_not(img)

# Display
cv2.imshow("Original", img)
cv2.imshow("Negative", negative)

cv2.waitKey(0)
cv2.destroyAllWindows()

#2.7

import cv2

# Read the image
img = cv2.imread("image.jpg")

# Mean Filter
mean = cv2.blur(img, (5,5))

# Gaussian Filter
gaussian = cv2.GaussianBlur(img, (5,5), 0)

# Median Filter
median = cv2.medianBlur(img, 5)

# Display images
cv2.imshow("Original", img)
cv2.imshow("Mean Filter", mean)
cv2.imshow("Gaussian Filter", gaussian)
cv2.imshow("Median Filter", median)

cv2.waitKey(0)
cv2.destroyAllWindows()

#2.8

import cv2

# Read image in grayscale
img = cv2.imread("image.jpg", 0)

# Gradient in X direction
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)

# Gradient in Y direction
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

cv2.imshow("Original", img)
cv2.imshow("Sobel X", sobelx)
cv2.imshow("Sobel Y", sobely)

cv2.waitKey(0)
cv2.destroyAllWindows()

#gradient magnitude

import cv2

img = cv2.imread("image.jpg", 0)

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

gradient = cv2.magnitude(sobelx, sobely)

cv2.imshow("Gradient", gradient)

cv2.waitKey(0)
cv2.destroyAllWindows()

#2.9

import cv2

img = cv2.imread("image.jpg", 0)

lap = cv2.Laplacian(img, cv2.CV_64F)

cv2.imshow("Original", img)
cv2.imshow("Laplacian", lap)

cv2.waitKey(0)
cv2.destroyAllWindows()


import cv2

# Read image in grayscale
img = cv2.imread("image.jpg", 0)

# Step 1: Gaussian Blur
gaussian = cv2.GaussianBlur(img, (5,5), 0)

# Step 2: Laplacian
log = cv2.Laplacian(gaussian, cv2.CV_64F)

# Display images
cv2.imshow("Original", img)
cv2.imshow("Gaussian Blur", gaussian)
cv2.imshow("Laplacian of Gaussian", log)

cv2.waitKey(0)
cv2.destroyAllWindows()

#2.10

cv2.Canny(img, 50, 150)

#edge detection using canny
import cv2

img = cv2.imread("image.jpg", 0)

edges = cv2.Canny(img, 50, 150)

cv2.imshow("Edges", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()

#line detection using hough transform

import cv2
import numpy as np

img = cv2.imread("image.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 150)

lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100,
                        minLineLength=50,
                        maxLineGap=10)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Lines", img)

cv2.waitKey(0)
cv2.destroyAllWindows()

#2.12 

import cv2

# Read two images
img1 = cv2.imread("image1.jpg")
img2 = cv2.imread("image2.jpg")

# Find absolute difference
difference = cv2.absdiff(img1, img2)

# Display images
cv2.imshow("Image 1", img1)
cv2.imshow("Image 2", img2)
cv2.imshow("Difference", difference)

cv2.waitKey(0)
cv2.destroyAllWindows()

#2.13
#M1

import cv2

# Read image
img = cv2.imread("image.jpg")

# Rotate image
rotate90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
rotate180 = cv2.rotate(img, cv2.ROTATE_180)
rotate270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

# Display images
cv2.imshow("Original", img)
cv2.imshow("90 Degree", rotate90)
cv2.imshow("180 Degree", rotate180)
cv2.imshow("270 Degree", rotate270)

cv2.waitKey(0)
cv2.destroyAllWindows()

#M2
import cv2

img = cv2.imread("image.jpg")

rows, cols = img.shape[:2]

# Rotation matrix (45 degrees)
matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)

rotated = cv2.warpAffine(img, matrix, (cols, rows))

cv2.imshow("Rotated Image", rotated)

cv2.waitKey(0)
cv2.destroyAllWindows()

# (cols/2, rows/2) → Rotate around the center of the image.
# 45 → Rotate by 45 degrees.
# 1 → Keep the image at its original size (no zoom).

# warpAffine() takes:

# the original image
# the instruction sheet (matrix)

# and creates a new rotated image.

#2.14
import cv2

# Read image in grayscale
img = cv2.imread("image.jpg", 0)

# Convert image into vector
vector = img.flatten()

# Print vector
print(vector)

# Print shape
print("Original Shape:", img.shape)
print("Vector Shape:", vector.shape)

#flatten() converts the 2D image into a 1D vector.

#2.15
# (Without ML/DL)
# The easiest and most common method is Thresholding.

import cv2

# Read image in grayscale
img = cv2.imread("image.jpg", 0)

# Apply threshold
ret, foreground = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Display images
cv2.imshow("Original", img)
cv2.imshow("Foreground", foreground)

cv2.waitKey(0)
cv2.destroyAllWindows()

#2.15
import cv2

# Read image in grayscale
img = cv2.imread("image.jpg", 0)

# Apply binary threshold
ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Display images
cv2.imshow("Original", img)
cv2.imshow("Foreground", thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.THRESH_BINARY

# Binary thresholding.
# Rule:
# Pixel > 127  → 255 (White)
# Pixel ≤127   → 0 (Black)

#2.16
import cv2
from matplotlib import pyplot as plt

# Read image in grayscale
img = cv2.imread("image.jpg", 0)

# Calculate histogram
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

# Display histogram
plt.plot(hist)
plt.title("Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")
plt.show()

#2.17
