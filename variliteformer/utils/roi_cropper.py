import cv2
import numpy as np
from PIL import Image


class ROICropper:
    """
    Adaptive ROI Cropper for Leukemia Cell Images.
    Focuses on darkest region (cell nucleus).
    """

    def __init__(self, crop_size=224):
        self.crop_size = crop_size

    def __call__(self, img):
        """
        img: PIL Image
        returns: Cropped PIL Image
        """

        # Convert PIL to OpenCV format
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Gaussian Blur to reduce noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold to detect dark region (nucleus)
        _, thresh = cv2.threshold(
            blur,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Find contours
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            # If no contour found, return original resized
            return img.resize((self.crop_size, self.crop_size))

        # Find largest contour (assumed cell)
        largest_contour = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(largest_contour)

        # Crop ROI
        roi = img_np[y:y+h, x:x+w]

        # Convert back to PIL
        roi_pil = Image.fromarray(roi)

        # Resize to fixed size
        roi_pil = roi_pil.resize((self.crop_size, self.crop_size))

        return roi_pil