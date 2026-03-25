import cv2
import numpy as np


def segment_nucleus(image):

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian blur
    blur = cv2.GaussianBlur(gray,(5,5),0)

    # Otsu threshold
    _,thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY+cv2.THRESH_OTSU
    )

    # Morphology cleanup
    kernel=np.ones((3,3),np.uint8)

    mask=cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel,
        iterations=2
    )

    # Apply mask
    segmented=cv2.bitwise_and(image,image,mask=mask)

    return segmented