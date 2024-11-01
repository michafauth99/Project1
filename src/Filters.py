from typing import Tuple
import numpy as np
from matplotlib import pyplot as plt


def median_filter(image: np.ndarray, filter_size: Tuple[int, int]) -> np.ndarray:
    """Returns an image after applying the median filter of the given size."""
    img_sz_x, img_sz_y = image.shape
    out_sz_x = img_sz_x - filter_size[0] + 1  # Why?
    out_sz_y = img_sz_y - filter_size[1] + 1  # Why?
    out = np.zeros(shape=(out_sz_x, out_sz_y), dtype=image.dtype)
    for i in range(out_sz_x):
        for j in range(out_sz_y):
            # YOUR CODE HERE:
            #   See `np.median(...)`.
            #   ...
            values = image[i:i + (filter_size[0]-1), j:j + (filter_size[1]-1)]
            out[i, j] = np.median(values)
    return out
