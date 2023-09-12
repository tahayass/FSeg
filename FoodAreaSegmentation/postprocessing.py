import cv2
import numpy as np



def open_mask(mask,kernel_size=(10,10)):
    # Opening is an operation that combines erosion followed by dilation. It's useful for removing noise and smoothing the mask.
    kernel = np.ones(kernel_size, np.uint8)
    opened_mask = cv2.dilate(cv2.erode(mask, kernel), kernel)
    return opened_mask


def close_mask(mask,kernel_size=(10,10)):
    # Closing is an operation that combines dilation followed by erosion. It's useful for closing small holes and gaps in the mask.
    kernel = np.ones(kernel_size, np.uint8)
    closed_mask = cv2.erode(cv2.dilate(mask, kernel), kernel)
    return closed_mask
