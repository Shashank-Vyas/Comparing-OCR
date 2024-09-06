import cv2
import numpy as np
import pytesseract
import re

# Load the image
image = cv2.imread('Video Images\\cframe618.jpg')
if image is None:
    raise ValueError("Image not found or unable to load.")

# Load the mask
mask = cv2.imread("Mask1.png", cv2.IMREAD_GRAYSCALE)
if mask is None:
    raise ValueError("Mask not found or unable to load.")
if mask.shape[:2] != image.shape[:2]:
    raise ValueError("Mask and image dimensions do not match.")

# Load the graphical overlay image
imgGraphics = cv2.imread("clog.png", cv2.IMREAD_UNCHANGED)  # Load with alpha channel if present

# Apply the mask to the image
masked_image = cv2.bitwise_and(image, image, mask=mask)

# Convert the masked image to grayscale
gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

# Invert the grayscale image
inverted = cv2.bitwise_not(gray)

# Threshold the inverted image
_, thresh = cv2.threshold(inverted, 150, 255, cv2.THRESH_BINARY_INV)

# Resize the masked image, thresholded ROI, and original image
resize_dimensions = (1200, 600)
resized_masked_image = cv2.resize(masked_image, resize_dimensions, interpolation=cv2.INTER_AREA)
resized_thresh = cv2.resize(thresh, resize_dimensions, interpolation=cv2.INTER_AREA)
resized_image = cv2.resize(image, resize_dimensions, interpolation=cv2.INTER_AREA)

# OCR on the resized ROI
text = pytesseract.image_to_string(resized_thresh)

print("OCR Output:")
print(text)

# Apply regular expressions to find the specific pattern
pattern = r'\d{1,3}\.\d{6},\s?\d{1,3}\.\d{6}'
matches = re.findall(pattern, text)

print("\nDetected matches:")
print(matches)

# Determine position to overlay the clog image (top-left corner)
x_offset, y_offset = 5, 5  # Adjust the position as needed

# Ensure the overlay fits within the resized image dimensions
x_end = x_offset + imgGraphics.shape[1]
y_end = y_offset + imgGraphics.shape[0]

if imgGraphics.shape[2] == 4:  # Check if the image has an alpha channel
    # Split the image into its channels
    b, g, r, a = cv2.split(imgGraphics)

    # Create a mask using the alpha channel and invert it
    overlay_mask = cv2.merge((b, g, r))
    alpha_channel = a / 255.0  # Normalize alpha channel to range [0, 1]

    # Blend the overlay image with the resized image using the alpha channel
    for c in range(0, 3):
        resized_image[y_offset:y_end, x_offset:x_end, c] = resized_image[y_offset:y_end, x_offset:x_end, c] * (1.0 - alpha_channel) + overlay_mask[:, :, c] * alpha_channel
else:
    # Simply overlay the clog image if there's no alpha channel
    resized_image[y_offset:y_end, x_offset:x_end] = imgGraphics

# Show the text on the image
if matches:
    for i, match in enumerate(matches):
        # Set the position and size for the background rectangle and text
        text_position = (50, 40 + i * 40)  # Position of the text (x, y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_size, _ = cv2.getTextSize(match, font, font_scale, font_thickness)
        background_top_left = (text_position[0] - 10, text_position[1] - 30)
        background_bottom_right = (text_position[0] + text_size[0] + 10, text_position[1] + 10)
        
        # Draw the solid background rectangle
        # cv2.rectangle(resized_image, background_top_left, background_bottom_right, (0, 0, 0), thickness=cv2.FILLED)
        
        # Put the GPS text on the image
        cv2.putText(resized_image, match, text_position, font, font_scale, (255, 255, 255), font_thickness)


# Display the images
# cv2.imshow("Resized Masked Image", resized_masked_image)
cv2.imshow("Resized Image with Overlay", resized_image)
# cv2.imshow("Resized Thresholded ROI", resized_thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
