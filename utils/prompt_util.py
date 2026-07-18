
import numpy as np
from sklearn.cluster import DBSCAN
def point_prompt(distance_map):
    max_points=150
    distance_map = np.array(distance_map)

    # ✅ Use adaptive threshold instead of fixed 0.2
    thresh_val = 0.1 * distance_map.max()
    binary_map = distance_map > thresh_val

    # 

    coords = np.column_stack(np.where(binary_map))

    if coords.shape[0] == 0:
        return []

    # ✅ Loosen clustering parameters
    db = DBSCAN(eps=4, min_samples=3).fit(coords)

    labels = db.labels_
    unique_labels = set(labels)

    centroids = []
    for label in unique_labels:
        if label == -1:
            continue
        cluster_points = coords[labels == label]
        centroid = cluster_points.mean(axis=0).astype(int)
        centroids.append(centroid)

    # ✅ Limit number of prompts
    if len(centroids) > max_points:
        centroids = centroids[:max_points]

    return centroids
