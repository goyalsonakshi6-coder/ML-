import cv2

# Read image
img = cv2.imread("WIN_20260616_17_25_13_Pro.jpg")
img =cv2.resize(img, (64 , 64))
# Check if image loaded
if img is None:
    print("Image not found!")
else:
    print("Image loaded successfully.")
    print("Shape:", img.shape)

    cv2.imshow("Original Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    