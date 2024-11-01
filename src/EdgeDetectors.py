import cv2
import numpy as np
from matplotlib import pyplot as plt

def find_edges(img_gray: np.ndarray) -> np.ndarray:
    """Adjust the Canny edge detector to find the edges in one image."""
    # YOUR CODE HERE:
    #   See `cv2.Canny(img_gray, ...)` with the `img_gray` as input.
    #   ...
    img_edges = cv2.Canny(img_gray, threshold1=100, threshold2=200, apertureSize=3)
    return img_edges


def find_corners(img_gray: np.ndarray) -> np.ndarray:
    """Adjust the Harris corner detector to find the corners in one image."""
    # YOUR CODE HERE: 
    #   See `cv2.cornerHarris(img_gray, ...)` with the `img_gray` as input.
    #   Use `cv2.dilate(...)` to remove non-local maxima.
    #   ...
    # Create a `corner` value for each pixel
    corner_map = cv2.cornerHarris(img_gray, blockSize=5, ksize=3, k=0.1)
    # Set img_corners[x, y] = 255 if (x, y) is local maxima, set to 0 otherwise
    corner_map_maxneighbourhood = cv2.dilate(corner_map, np.ones((11, 11)))
    img_corners = np.zeros_like(img_gray)
    # Return value for visualization, whose pixels should be either 0 o 255.
    img_corners[corner_map == corner_map_maxneighbourhood] = 255
    return img_corners