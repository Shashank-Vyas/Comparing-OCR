import cv2
import numpy as np
import pytesseract
import re

image = cv2.imread('cframe2681.png')
if image is None:
    raise ValueError("Image not found or unable to load.")


mask = cv2.imread("Mask1.png", cv2.IMREAD_GRAYSCALE)
if mask is None:
    raise ValueError("Mask not found or unable to load.")
if mask.shape[:2] != image.shape[:2]:
    raise ValueError("Mask and image dimensions do not match.")

# Apply the mask to the image
masked_image = cv2.bitwise_and(image, image, mask=mask)

gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

inverted = cv2.bitwise_not(gray)

_, thresh = cv2.threshold(inverted, 150, 255, cv2.THRESH_BINARY_INV)

# Resize both masked image and thresholded ROI
resize_dimensions = (1200, 600)  
resized_masked_image = cv2.resize(masked_image, resize_dimensions, interpolation=cv2.INTER_AREA)
resized_thresh = cv2.resize(thresh, resize_dimensions, interpolation=cv2.INTER_AREA)

# OCR on the resized ROI
text = pytesseract.image_to_string(resized_thresh)

print("OCR Output:")
print(text)

# Apply regular expressions to find the specific pattern
pattern = r'\d{1,3}\.\d{6},\s?\d{1,3}\.\d{6}'
matches = re.findall(pattern, text)

print("\nDetected matches:")
print(matches)

cv2.imshow("Resized Masked Image", resized_masked_image)
cv2.imshow("Resized Thresholded ROI", resized_thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
