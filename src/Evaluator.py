
# Class which evaluates the performance of the methods, it calculates the FP, FN, TP, TN, Precision, Recall, F1-Score 
import numpy as np
from scipy.spatial.distance import cdist

import pandas as pd

class Evaluator:
    def __init__(self):
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.TP_boxes = []
        self.TP_points = []
        self.FP_boxes = []
        self.FN_points = []
        
    
    def non_max_suppression(self, predictions, radius):
        """
        Apply NMS to a list of points based on a distance threshold.
        :param points: np array of points as (x, y, confidence).
        :param radius: Distance threshold for suppression.
        :return: List of points after NMS.
        """
        if len(predictions) == 0:
            return predictions
        
        # Sort by confidence descending
        predictions = predictions[np.argsort(predictions[:, 2])[::-1]]
        retained = []
        suppressed = np.zeros(len(predictions), dtype=bool)

        for i, (x1, y1, conf1) in enumerate(predictions):
            if suppressed[i]:
                continue
            retained.append((x1, y1, conf1))
            for j, (x2, y2, conf2) in enumerate(predictions):
                if j <= i or suppressed[j]:
                    continue
                if np.sqrt((x1 - x2)**2 + (y1 - y2)**2) <= radius:
                    suppressed[j] = True

        return np.array(retained)

    def evaluate(self, preds, actual, radius=10, nms_radius=10):
        """
        Evaluate the performance of a method based on the predictions and actual points.
        :param predictions: List of predicted points as (x, y).
        :param actual: List of actual points as (x, y).
        :param radius: Distance threshold for a prediction to be considered correct.
        """

        if len(preds) == 0:
            self.FN = len(actual)
            self.TP = 0 
            self.FP = 0
            return
        
        if len(actual) == 0:
            self.FN = 0
            self.TP = 0
            self.FP = len(preds)
            return
        
        # Apply NMS to predictions
        preds = self.non_max_suppression(preds, nms_radius)
        
        # Remove last dimension in predictions
        pred_coords = preds[:, :2]
        
        # Compute distance matrix
        dists = cdist(pred_coords, actual, metric='euclidean')
          
        matched_actual = set()
        matched_pred = set()

        # Find the closest actual point for each prediction
        for pred_idx, distances in enumerate(dists):
            for actual_idx, distance in enumerate(distances):
                if distance < radius and actual_idx not in matched_actual:
                    matched_actual.add(actual_idx)
                    matched_pred.add(pred_idx)
                    break
        self.TP = len(matched_actual)
        self.FP = len(preds) - self.TP
        self.FN = len(actual) - self.TP


    def evaluate_bounding_boxes(self, pred_boxes, actual):
        """
        Evaluate the performance of a method based on the predictions and actual points.
        :param predictions: List of predicted points as (x, y).
        :param actual: List of actual points as (x, y).
        :param radius: Distance threshold for a prediction to be considered correct.
        """

        # Initialize counters
        self.TP = 0
        self.FP = 0
        self.FN = 0

        if len(pred_boxes) == 0:
            # No predictions, all actual points are missed
            self.FN = len(actual)
            self.FN_points = actual
            return

        if len(actual) == 0:
            # No ground truth, all predicted boxes are false positives
            self.FP = len(pred_boxes)
            self.FP_boxes = pred_boxes
            return

        # Track matched points to prevent duplicate TP counting
        matched_points = set()
        
    

        # Track matched predicted boxes
        matched_pred_boxes = set()

        # Evaluate True Positives
        for pred_box in pred_boxes:
            # Convert (x, y, w, h) to (x1, y1, x2, y2)
            x1, y1, w, h = pred_box
            x2 = x1 + w
            y2 = y1 + h

            for i, actual_point in enumerate(actual):
                if i not in matched_points:  # Skip already matched points
                    if (
                        x1 - 5 <= actual_point[0] <= x2 + 5 and  # x within box
                        y1 - 10 <= actual_point[1] <= y2       # y within box
                    ):
                        self.TP += 1
                        self.TP_boxes.append(pred_box)
                        self.TP_points.append(actual_point)
                        matched_points.add(i)          # Mark this point as matched
                        matched_pred_boxes.add(tuple(pred_box))  # Mark this box as matched
                        #break  # Stop checking other points for this box

        # Calculate False Positives (predictions not matching any ground truth)
        self.FP_boxes = [box for box in pred_boxes if tuple(box) not in matched_pred_boxes]
        self.FP = len(self.FP_boxes)

        # Calculate False Negatives (ground truth points not matched by any prediction)
        self.FN_points = [point for i, point in enumerate(actual) if i not in matched_points]
        self.FN = len(self.FN_points)
                    

    def get_precision(self):
        return self.TP / (self.TP + self.FP) if self.TP + self.FP > 0 else 1.0

    def get_recall(self):
        return self.TP / (self.TP + self.FN) if self.TP + self.FN > 0 else 1.0

    def get_f1_score(self):
        precision = self.get_precision()
        recall = self.get_recall()
        if precision == 1 and recall == 1:
            return 1.0
        return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    def get_RMSE(self, actual_nr, pred_nr):
        return np.sqrt((actual_nr - pred_nr) ** 2)
    
    def get_accuracy(self):
        return (self.TP / (self.TP  + self.FP))*100 if self.TP  + self.FP > 0 else 100

    def get_confusion_matrix(self):
       
        # Construct the confusion matrix as a NumPy array
        cm = np.array([[self.TP, self.FN], [self.FP, 0]])  # 0 for TN (Not being counted here)

        
        # Construct the confusion matrix as a DataFrame
        cm_df = pd.DataFrame(cm, columns=['Predicted Positive', 'Predicted Negative'], index=['Actual Positive', 'Actual Negative'])
        
        return cm_df

        

    def get_metrics(self):
        return self.get_precision(), self.get_recall(), self.get_f1_score(), self.get_accuracy()
    
    def get_bounding_boxes(self):
        return self.TP_boxes, self.TP_points, self.FP_boxes, self.FN_points
    

    def reset(self):
        self.TP = 0
        self.FP = 0
        self.FN = 0
        



