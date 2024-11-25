
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt



def preprocess_image(image: np.ndarray, noise_reduction="Gaussian",filter_size=5,CLAHE=True, grid_size=8) -> np.ndarray:
    """
    Preprocesses an image by ...
    """
    # Convert image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Noise Reduction
    if noise_reduction == "Gaussian":
        image = cv2.GaussianBlur(image, (filter_size, filter_size), 0)
    elif noise_reduction == "Median":
        image = cv2.medianBlur(image, filter_size)

    # Apply CLAHE
    if CLAHE:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(grid_size,  grid_size))
        image = clahe.apply(image)

    return image


def substract_empty_beach(image):
    # load empty beach image
    empty_beach = cv2.imread("data/images/0_empty.jpg")
    # convert to binary image with histeresis method
    empty_beach = cv2.cvtColor(empty_beach, cv2.COLOR_BGR2GRAY)
    #
    # apply clahe to improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    empty_beach = clahe.apply(empty_beach)
    empty_beach = cv2.GaussianBlur(empty_beach, (5, 5), 0)
    empty_beach = binarize_by_hysteresis(empty_beach, 80, 150)

    # apply erosion to remove noise
    kernel = np.ones((5, 5), np.uint8)
    empty_beach = cv2.erode(empty_beach, kernel, iterations=1)
    image_substracted = np.where((image == 0) & (empty_beach == 0), 255, image)

    return image_substracted

def background_subtraction(image, pct=35,static_foreground_removal=False):
    """
    Apply the moving average algorithm with boustrophedon scanning to a grayscale image.

    Parameters:
        image (numpy.ndarray): The input grayscale image as a 2D array.
        pct (float): Percentage threshold for deciding black or white pixel.

    Returns:
        numpy.ndarray: A binary image (2D array) with pixels set to black or white.
    """
    # Ensure input is a 2D grayscale image
    if len(image.shape) != 2:
        raise ValueError("Input image must be a 2D grayscale image.")

    # Dimensions of the image
    rows, cols = image.shape

    # Initialize moving average to 127 * n
    n = cols // 8
    M = 127 * n

    # Create output binary image
    output_image = np.zeros_like(image, dtype=np.uint8)

    # Define threshold multiplier based on percentage
    threshold_multiplier = (100 - pct) / 100

    # Implement boustrophedon scanning
    for i in range(rows):
        # Determine the direction of scanning (normal or reversed)
        if i % 2 == 0:  # Even row: left to right
            col_range = range(cols)
        else:  # Odd row: right to left
            col_range = range(cols - 1, -1, -1)

        for j in col_range:
            # Get the pixel intensity
            g_ij = image[i, j]

            # Compute the threshold
            threshold = (M / n) * threshold_multiplier

            # Apply threshold to determine binary output
            if g_ij < threshold:
                output_image[i, j] = 0  # Set to black
            else:
                output_image[i, j] = 255  # Set to white

            # Update moving average
            M = M - (M / n) + g_ij
        
        # remove mountain background by location
        
        output_image[0:450,0:1000]=255
        output_image[0:420,1000:1920]=255

        # remove static foreground by substracting empty beach image
    if static_foreground_removal:
        
        output_image = substract_empty_beach(output_image)

    return cv2.bitwise_not(output_image)

def apply_morphological_operations(image, dilation=False, opening=False, closing=False, smooth=False):
    """
    Apply morphological operations to the image.

    Parameters:
        dilation (bool): Apply dilation operation.
        opening (bool): Apply opening operation.
        closing (bool): Apply closing operation.

    Returns:
        numpy.ndarray: The processed image.
    """

    if dilation:  
        kernel_dilate = np.ones((3, 3), np.uint8)
        roi_dilate = image[438:600, :]

        # Apply dilation to fill in the gaps
        roi_dilated = cv2.dilate(roi_dilate, kernel_dilate, iterations=1)
        image[438:600, :] = roi_dilated

    if opening:
        kernel_open = np.ones((5, 5), np.uint8)
        roi_open = image[700:1080, :]

        # Apply opening to remove noise
        roi_opened = cv2.morphologyEx(roi_open, cv2.MORPH_OPEN, kernel_open)
        image[700:1080, :] = roi_opened

    if closing:
        kernel_close = np.ones((5, 5), np.uint8)
        # Apply closing to fill in the gaps
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_close)
    
    if smooth:
        # Apply gaussian blur to smooth the image
        image = cv2.GaussianBlur(image, (3, 3), 0)
    
    return image

def non_max_suppression(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    # compute the area of the bounding boxes
    area = boxes[:, 2] * boxes[:, 3]
    # sort the bounding boxes by the bottom-right y-coordinate of the bounding box
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # find the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have sufficient overlap
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked
    return boxes[pick]

def post_process_image(image, stats, centroids, nms_threshold=0.5):

    df = pd.DataFrame(stats, columns=['left', 'top', 'width', 'height', 'area'])
    df = df.iloc[1:]  # Remove the first row (background)
    centroids= centroids[1:]  # Remove the first row (background)

    points = [(0, 603), (129, 593), (249, 582), (339, 577) , (425, 573), (538, 566), (666, 560),(784, 551), (902, 542), (1056, 534), (1189, 522), (1292, 513), (1403, 502), (1488, 483), (1558, 470), (1610, 455), (1625, 440), (1614, 425)]

    # fit a curve to the points
    shoreline = np.poly1d(np.polyfit([p[0] for p in points], [p[1] for p in points], 2))

    # Filter out small objects  if the area is less than 80 pixels and the centroid is below the curve above the curve the objects can be smaller
    df = df[
    ((df['area'] > 80) | (centroids[:, 1] < shoreline(centroids[:, 0])))  # Keep small objects above the shoreline
    & ~((df['area'] > 500) & (centroids[:, 1] < shoreline(centroids[:, 0])))  # Filter out large objects above the shoreline
    ]
    
    # Filter out objects where the ratio of hight to width is greater than 5
    df = df[((df['height'] / df['width']) < 8) & ((df['width'] / df['height']) < 2)]

    # Filter out big objects where either the width or height is greater than 150 pixels
    df = df[(df['width'] < 150) & (df['height'] < 150)]

    # Remove static foreground in lower left and right corners
    df = df[~((df['top'] > 900) & (df['left'] < 125))]
    df = df[~((df['top'] > 750) & (df['left'] > 1800))]

    # Non maximum suppression
    boxes = df[['left', 'top', 'width', 'height']].values
    boxes = non_max_suppression(boxes, nms_threshold)

    return boxes

def draw_result(image, tp_boxes, tp_points, fp_boxes, fn_points):
    image_result = image.copy()

    # convert image to BGR
    image_result = cv2.cvtColor(image_result, cv2.COLOR_RGB2BGR)
        
    for x, y, w, h in tp_boxes:
        cv2.rectangle(image_result, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for x, y, w, h in fp_boxes:
        cv2.rectangle(image_result, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    for x, y in tp_points:
        cv2.circle(image_result, (x, y), 3, (0, 255, 0), 2)

    for x, y in fn_points:
        cv2.circle(image_result, (x, y), 3, (0, 0, 255), 2)

    return image_result

def print_evaluation(eval, actual_persons):
    conf_matrix = eval.get_confusion_matrix()
    precision = eval.get_precision()
    recall = eval.get_recall()
    f1 = eval.get_f1_score()
    rmse = eval.get_RMSE(actual_persons, eval.TP)

    # print eval results with line brake
    print("confusion matrix: \n", conf_matrix)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"RMSE: {rmse:.4f}")
        

        

    
    
