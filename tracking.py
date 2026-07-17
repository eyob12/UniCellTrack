import numpy as np
import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
import cv2
import networkx as nx
import time as timing

# ---------------------------- Visualization Function ----------------------------
def visualize_tracking(seg_img, raw_img, title, frame_index, save_dir=None):
    plt.figure(figsize=(10, 10))
    if raw_img is not None:
        plt.imshow(raw_img, cmap='gray')
    unique_labels = np.unique(seg_img)
    unique_labels = unique_labels[unique_labels != 0]
    if len(unique_labels) > 0:
        cmap = plt.cm.tab20 if len(unique_labels) <= 20 else plt.cm.get_cmap('nipy_spectral', len(unique_labels))
        color_seg = np.zeros((*seg_img.shape, 3))
        for idx, label in enumerate(unique_labels):
            mask = (seg_img == label)
            color_seg[mask] = cmap(idx % cmap.N)[:3]
        plt.imshow(color_seg, alpha=0.5)
    centers = cell_center(seg_img)
    for label, center in centers.items():
        plt.text(center[1], center[0], str(label), color='yellow', fontsize=12, ha='center', va='center')
    plt.title(f"{title} - Frame {frame_index}")
    plt.axis('off')
    # plt.show()

        # Save image
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{title}_frame{frame_index:03d}.png")
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.close()  # avoid showing every figure in loop

# ---------------------------- Utility Functions ----------------------------
def cell_center(seg_img):
    results = {}
    for label in np.unique(seg_img):
        if label != 0:
            ys, xs = np.where(seg_img == label)
            results[label] = [np.round(np.mean(ys)), np.round(np.mean(xs))]
    return results

def resize_mask_to_shape(mask, target_shape):
    return cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

def compute_cell_location(seg_img):
    g = nx.Graph()
    centers = cell_center(seg_img)
    labels = [l for l in np.unique(seg_img) if l != 0]
    for i, li in enumerate(labels):
        g.add_node(li)
        for lj in labels[i+1:]:
            pi, pj = centers[li], centers[lj]
            dist = np.sqrt((pi[0]-pj[0])**2 + (pi[1]-pj[1])**2)
            g.add_edge(li, lj, weight=dist)
    return g


def knn_graph(centers, k=5):
    """centers: dict{id -> [y,x]} -> (neighbors, mean_kdist)"""
    ids = list(centers.keys())
    if len(ids) <= 1:
        return {i: [] for i in ids}, {i: np.inf for i in ids}
    pts = np.array([centers[i] for i in ids], dtype=float)
    D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    k_eff = min(k, max(1, len(ids) - 1))
    nbr_idx = np.argpartition(D, k_eff, axis=1)[:, :k_eff]
    neighbors = {ids[i]: [ids[j] for j in nbr_idx[i]] for i in range(len(ids))}
    mean_k = {ids[i]: float(np.mean(D[i, nbr_idx[i]])) for i in range(len(ids))}
    return neighbors, mean_k

def topo_penalty(cid1, cid2, c1, c2, neighbors, assigned_prev2curr):
    """Mean neighbor-distance distortion for already-assigned neighbors."""
    nbrs = neighbors.get(cid1, [])
    if not nbrs:
        return 0.0
    p1 = np.array(c1[cid1], dtype=float)
    q2 = np.array(c2[cid2], dtype=float)
    pen, cnt = 0.0, 0
    for nb in nbrs:
        if nb in assigned_prev2curr:              # only use neighbors already matched
            j_nb = assigned_prev2curr[nb]
            p_nb = np.array(c1[nb], dtype=float)
            q_nb = np.array(c2[j_nb], dtype=float)
            pen += abs(np.linalg.norm(q2 - q_nb) - np.linalg.norm(p1 - p_nb))
            cnt += 1
    return pen / (cnt + 1e-6)

def smooth_flows_by_graph(flows, neighbors, lam=0.5):
    """Optional: flow smoothing with neighbor averaging."""
    if flows is None:
        return None
    sm = {}
    for i, v in flows.items():
        nbv = [flows[n] for n in neighbors.get(i, []) if n in flows]
        if nbv:
            nbm = np.mean(np.array(nbv, dtype=float), axis=0)
            sm[i] = ((v[0] + lam * nbm[0]) / (1 + lam), (v[1] + lam * nbm[1]) / (1 + lam))
        else:
            sm[i] = v
    return sm
import numpy as np

def tracklet(
    g1, g2, seg_img1, seg_img2, maxtrackid, frame, linelist,
    flows=None, k_nn=5, gamma=2.0, beta=0.1, use_flow_smoothing=True
):
    """
    Enhanced tracklet association with normalized cost terms so topology penalty
    has comparable influence to spatial distance.

    Parameters:
        k_nn: neighbors per node for topology
        gamma: density gate scale
        beta: topology penalty weight
        use_flow_smoothing: smooth flows using kNN in prev frame
    """
    f1, f2 = {}, {}
    c1, c2 = cell_center(seg_img1), cell_center(seg_img2)
    for v in g1.degree(weight='weight'):
        f1[v[0]] = [c1[v[0]], v[1]]
    for v in g2.degree(weight='weight'):
        f2[v[0]] = [c2[v[0]], v[1]]

    # Build kNN graph (prev frame) and density scale
    neighbors_prev, mean_k_prev = knn_graph(c1, k=k_nn)

    # Optional flow smoothing
    if use_flow_smoothing and flows is not None:
        flows = smooth_flows_by_graph(flows, neighbors_prev, lam=0.5)

    mapping, inv_mapping = {}, {}
    assigned_prev2curr = {}

    # For normalization: collect distances and penalties first
    dist_samples, pen_samples = [], []

    # First pass: compute all distances and penalties for scaling
    tmp_cost_data = {}
    for cid2 in f2:
        tmp_cost_data[cid2] = []
        for cid1 in f1:
            # Predicted position with flow
            if flows is not None and cid1 in flows:
                pred_y = f1[cid1][0][0] + flows[cid1][0]
                pred_x = f1[cid1][0][1] + flows[cid1][1]
                dy = f2[cid2][0][0] - pred_y
                dx = f2[cid2][0][1] - pred_x
            else:
                dy = f2[cid2][0][0] - f1[cid1][0][0]
                dx = f2[cid2][0][1] - f1[cid1][0][1]
            base_d2 = dy*dy + dx*dx
            pen = topo_penalty(cid1, cid2, c1, c2, neighbors_prev, assigned_prev2curr)
            dist_samples.append(np.sqrt(base_d2))
            pen_samples.append(pen)
            tmp_cost_data[cid2].append((cid1, base_d2, pen))

    # Normalization factors (avoid zero by adding small eps)
    dist_scale = np.std(dist_samples) + 1e-6
    pen_scale = np.std(pen_samples) + 1e-6

    # Second pass: assign with normalized costs
    for cid2 in tmp_cost_data:
        best_cost, best_match = float('inf'), None
        best_base_d2 = float('inf')

        for cid1, base_d2, pen in tmp_cost_data[cid2]:
            norm_dist = np.sqrt(base_d2) / dist_scale
            norm_pen = pen / pen_scale
            cost = norm_dist + beta * (norm_pen ** 2)  # normalized cost

            if cost < best_cost:
                best_cost, best_match = cost, cid1
                best_base_d2 = np.sqrt(base_d2)

        tau = gamma * mean_k_prev.get(best_match, np.inf)
        if best_match is not None and (best_base_d2 <= tau):
            mapping[cid2] = best_match
            inv_mapping.setdefault(best_match, []).append(cid2)
            assigned_prev2curr[best_match] = cid2

    # Apply mapping results
    new_seg_img2 = np.zeros_like(seg_img2)

    # Continuations and Mitosis
    for pid, children in inv_mapping.items():
        if len(children) == 1:
            cid = children[0]
            new_seg_img2[seg_img2 == cid] = pid
            for i, line in enumerate(linelist):
                parts = line.split()
                if int(parts[0]) == pid:
                    linelist[i] = f"{pid} {parts[1]} {frame+1} {parts[3]}"
                    break
        else:
            for i, line in enumerate(linelist):
                parts = line.split()
                if int(parts[0]) == pid:
                    linelist[i] = f"{pid} {parts[1]} {frame} {parts[3]}"
                    break
            for cid in children:
                maxtrackid += 1
                new_seg_img2[seg_img2 == cid] = maxtrackid
                linelist.append(f"{maxtrackid} {frame+1} {frame+1} {pid}")

    # New appearances
    matched_cids2 = set(mapping.keys())
    for cid2 in f2:
        if cid2 not in matched_cids2:
            maxtrackid += 1
            new_seg_img2[seg_img2 == cid2] = maxtrackid
            linelist.append(f"{maxtrackid} {frame+1} {frame+1} 0")

    return maxtrackid, linelist, new_seg_img2

# ---------------------------- Main Tracking Function ----------------------------
def track_main(seg_fold, raw_fold, track_fold, ds, section, visualize=False):
    if not os.path.exists(track_fold):
        os.makedirs(track_fold)
    seg_files = sorted([f for f in os.listdir(seg_fold) if f.endswith('.tif')])
    maxtrackid = 0
    linelist = []
    threshold = 20
    raw_prev = None
    total_start_time = timing.time()
    for frame in range(len(seg_files)):
        start_time = timing.time()
        seg_path = os.path.join(seg_fold, seg_files[frame])
        if frame == 0:
            # Initialization for first frame
            img_arr = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
            # Remove small objects
            labels, counts = np.unique(img_arr, return_counts=True)
            for label, count in zip(labels, counts):
                if count < threshold and label != 0:
                    img_arr[img_arr == label] = 0
            # Relabel sequentially
            new_labels = {old: new for new, old in enumerate(np.unique(img_arr))}
            img_arr_relabel = np.vectorize(new_labels.get)(img_arr)
            # Save initial tracked mask
            save_name = f"mask{frame:04d}.tif" if ds in ['BF-C2DL-MuSC','BF-C2DL-HSC'] else f"mask{frame:03d}.tif"
            sitk.WriteImage(sitk.GetImageFromArray(img_arr_relabel.astype('uint16')), os.path.join(track_fold, save_name))
            # Initialize tracking list (track_id, start_frame, end_frame, parent_id)
            for label in np.unique(img_arr_relabel):
                if label != 0:
                    linelist.append(f"{label} {frame} {frame} 0")
                    maxtrackid = max(maxtrackid, label)
            # Read raw image for first frame (for optical flow use in next step)
            raw_path = os.path.join(raw_fold, f"t{frame:04d}.tif") if ds in ['BF-C2DL-MuSC','BF-C2DL-HSC'] else os.path.join(raw_fold, f"t{frame:03d}.tif")
            raw_prev = sitk.GetArrayFromImage(sitk.ReadImage(raw_path))
            continue

        # Read previous tracked mask and current segmentation
        prev_mask_name = f"mask{frame-1:04d}.tif" if ds in ['BF-C2DL-MuSC','BF-C2DL-HSC'] else f"mask{frame-1:03d}.tif"
        prev_img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(track_fold, prev_mask_name)))
        curr_img = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
        # Remove small objects in current segmentation
        labels, counts = np.unique(curr_img, return_counts=True)
        for label, count in zip(labels, counts):
            if count < threshold and label != 0:
                curr_img[curr_img == label] = 0
        # Read current raw image
        raw_path = os.path.join(raw_fold, f"t{frame:04d}.tif") if ds in ['BF-C2DL-MuSC','BF-C2DL-HSC'] else os.path.join(raw_fold, f"t{frame:03d}.tif")
        raw_curr = sitk.GetArrayFromImage(sitk.ReadImage(raw_path))
        # Prepare images for optical flow (convert to 8-bit single channel)
        def prepare_for_flow(img):
            if img.dtype != np.uint8:
                img_norm = img.astype(np.float32)
                if img_norm.max() > 0:
                    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min()) * 255.0
                else:
                    img_norm = np.zeros_like(img_norm)
                return img_norm.astype(np.uint8)
            else:
                return img
        prev_flow_img = prepare_for_flow(raw_prev)
        curr_flow_img = prepare_for_flow(raw_curr)
        # Compute dense optical flow between previous and current raw frames
        flow = cv2.calcOpticalFlowFarneback(prev_flow_img, curr_flow_img, None,
                                           0.5, 3, 15, 3, 5, 1.2, 0)
        # Compute average flow vector for each cell in prev_img
        flows_dict = {}
        h_seg, w_seg = prev_img.shape
        h_raw, w_raw = flow.shape[:2]
        y_scale = h_raw / float(h_seg)
        x_scale = w_raw / float(w_seg)
        for label in np.unique(prev_img):
            if label == 0: 
                continue
            ys, xs = np.where(prev_img == label)
            if ys.size == 0:
                continue
            if h_seg != h_raw or w_seg != w_raw:
                # Map segmentation coordinates to raw image coordinates
                raw_ys = np.round(ys * y_scale).astype(int)
                raw_xs = np.round(xs * x_scale).astype(int)
                raw_ys = np.clip(raw_ys, 0, h_raw - 1)
                raw_xs = np.clip(raw_xs, 0, w_raw - 1)
                flow_vals = flow[raw_ys, raw_xs]
            else:
                flow_vals = flow[ys, xs]
            avg_flow_y = np.mean(flow_vals[:, 1]) if flow_vals.size > 0 else 0.0
            avg_flow_x = np.mean(flow_vals[:, 0]) if flow_vals.size > 0 else 0.0
            # Convert flow from raw pixels to segmentation pixel units
            flow_seg_y = avg_flow_y / y_scale
            flow_seg_x = avg_flow_x / x_scale
            flows_dict[label] = (flow_seg_y, flow_seg_x)
        # Compute cell location graphs for tracking
        g1 = compute_cell_location(prev_img)
        g2 = compute_cell_location(curr_img)
        # Perform tracking (matching and assigning IDs) using optical flow predictions
        # maxtrackid, linelist, tracked_img = tracklet(g1, g2, prev_img, curr_img,
        #                                             maxtrackid, frame - 1, linelist,
        #                                             flows=flows_dict)
        maxtrackid, linelist, tracked_img = tracklet(
                                            g1, g2, prev_img, curr_img,
                                            maxtrackid, frame - 1, linelist,
                                            flows=flows_dict,       # from your Farnebäck step
                                            k_nn=5,                 # 3–7 is typical
                                            gamma=2.0,              # density gate scale (1.5–2.5)
                                            beta=0.1,               # topology penalty weight (0.05–0.3)
                                            use_flow_smoothing=True # try True; set False if you prefer raw flow
                                        )
        # Save the tracked mask for current frame
        save_name = f"mask{frame:04d}.tif" if ds in ['BF-C2DL-MuSC','BF-C2DL-HSC'] else f"mask{frame:03d}.tif"
        sitk.WriteImage(sitk.GetImageFromArray(tracked_img.astype('uint16')), os.path.join(track_fold, save_name))
        # Visualization if required
        if visualize:
            raw_img = raw_curr
            resized_mask = resize_mask_to_shape(tracked_img, raw_img.shape)
            vis_dir = os.path.join(track_fold, "visualizations")
            visualize_tracking(resized_mask, raw_img, f"{section}_{ds}", frame, save_dir=vis_dir)

            # visualize_tracking(resized_mask, raw_img, f"{section}_{ds}", frame)
        # Prepare for next iteration
        raw_prev = raw_curr

    # Save tracking results to text file
    with open(os.path.join(track_fold, 'res_track.txt'), 'w') as f:
        for line in linelist:
            f.write(line + '\n')
    print(f'Total processing time: {timing.time() - total_start_time:.2f} seconds')

# -------------- Example Usage --------------
if __name__ == "__main__":
    datasets = [ 
                'PhC-C2DH-U373',
                'PhC-C2DL-PSC', 
                'DIC-C2DH-HeLa',
                'Fluo-C2DL-MSC',
                'Fluo-N2DH-GOWT1', 
                'Fluo-N2DL-HeLa',
                'Fluo-N2DH-SIM+', 
                'BF-C2DL-HSC',
                'BF-C2DL-MuSC'
]
    for ds in datasets:
        for section in ['01', '02']:
            # if section=='02':
            #     continue
            seg_fold = f"./seg/{ds}/{section}_{ds}_test/Instance_mask"  # seg result
            raw_fold= f"./Test_datasets/{ds}/{section}/" # raw image 

            track_fold = f"./tracking/{section}_{ds}_test/{section}_RES" # track save dir
            print(f"Running--{section}_{ds.replace('-', '_')}")
            track_main(seg_fold, raw_fold, track_fold, ds, section, visualize=True)
            
          
        