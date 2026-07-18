import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage import morphology
from scipy import ndimage
import skimage.io as io

def post_processing(predicted_mask):
    binary_mask = (predicted_mask > 0.1).astype(np.uint8)
    
    # If the binary mask is too small, label and return it
    if np.sum(binary_mask) < 20:
        labeled_mask, _ = ndimage.label(binary_mask)  # Get only the labeled mask
        return labeled_mask  # Return only the labeled mask
    
    marker = (predicted_mask > 0.1).astype(np.uint8)

    # Compute the distance transform
    dist_transform = cv2.distanceTransform(marker, cv2.DIST_L2, 5)
    distance = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX) 
    
    # Threshold to get peaks
    _, peaks = cv2.threshold(distance, 0.7 * dist_transform.max(), 255, 0)

    # Find markers
    peaks = np.uint8(peaks)
    _, markers = cv2.connectedComponents(peaks)

    # Apply watershed
    markers = cv2.watershed(cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR), markers)

    # Detect local maxima in the distance transform
    local_max = peak_local_max(
        distance,
        footprint=np.ones((3, 3)),
        min_distance=5,
        threshold_abs=0.1
    )
    markers = ndimage.label(local_max)[0]

    # Fallback: Force markers if none are found
    if np.max(markers) < 2:
        markers = (distance > 0.1 * distance.max()).astype(np.uint8)
        markers = ndimage.label(markers)[0]

    # Apply Watershed Algorithm
    ws_labels = watershed(-distance, markers, mask=binary_mask)

    # Remove small objects (clean up noise)
    ws_labels = morphology.remove_small_objects(ws_labels, min_size=20)
    
    # If no labels were found (empty mask), label binary_mask
    if ws_labels.max() == 0:
        labeled_mask, _ = ndimage.label(binary_mask)  # Get only the labeled mask
        return labeled_mask  # Return only the labeled mask

    return ws_labels  # Ensure this is a NumPy array
