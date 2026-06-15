import os
import glob
import yaml
import argparse
from typing import List, Dict, Tuple
from collections import Counter

import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
from tqdm import tqdm

from utils.path import basename, filename_wo_ext
from utils.fileio import str_to_tuple

def load_binary_masks_as_dict(masks_dir) -> Dict[int, np.ndarray]:
    mask_paths = glob.glob(os.path.join(masks_dir, "*.png"))
    masks = [cv2.imread(mask_path, -1) for mask_path in mask_paths]
    mask_ids = [int(filename_wo_ext(mask_path)) for mask_path in mask_paths]
    return {mask_id: mask for mask_id, mask in zip(mask_ids, masks)}

# get_leaf_root_wls (OpenCV equivalent)
def get_leaf_root_opencv(masks_dir: str) -> Tuple[np.ndarray, List[int]]:
    mask_paths = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
    clustering_points = []
    mask_ids = []

    for mask_path in mask_paths:
        try:
            mask_id = int(filename_wo_ext(mask_path))
        except ValueError:
            continue  

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None or np.count_nonzero(mask) < 50:
            continue

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)

        if len(cnt) < 5:
            continue

        (cx, cy), (MA, ma), angle = cv2.fitEllipse(cnt)
        angle_rad = np.deg2rad(angle)
        dx = np.sin(angle_rad)
        dy = -np.cos(angle_rad)

        cnt_squeeze = cnt.squeeze(axis=1) 
        centered_cnt = cnt_squeeze - np.array([cx, cy])
        projections = centered_cnt.dot(np.array([dx, dy]))

        min_idx = np.argmin(projections)
        max_idx = np.argmax(projections)

        pt1 = cnt_squeeze[min_idx]
        pt2 = cnt_squeeze[max_idx]

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            com_x = M["m10"] / M["m00"]
            com_y = M["m01"] / M["m00"]
            dist1 = np.hypot(pt1[0] - com_x, pt1[1] - com_y)
            dist2 = np.hypot(pt2[0] - com_x, pt2[1] - com_y)

            # Assign base point and store in [y, x] to match original logic
            base_pt = pt1 if dist1 > dist2 else pt2
            clustering_points.append([int(base_pt[1]), int(base_pt[0])])
            mask_ids.append(mask_id)

    return np.array(clustering_points), mask_ids

def greedyDBSCANclustering(clustering_points, eps: float, min_samples: int, num_steps: int=10):
    clustering_result = np.array([-1] * len(clustering_points))
    unclustered_idx_list = np.array(list(range(len(clustering_points)))) 
    unclustered_points = np.array(clustering_points.copy())
    
    for i in range(num_steps):
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(unclustered_points)
        labels = clustering.labels_
        
        outlier_removed_labels = labels[np.where(labels!=-1)]
        if len(outlier_removed_labels) == 0: 
            break
        else:
            counter = Counter(outlier_removed_labels)
            largest_cluster_label = counter.most_common(1)[0][0]
            
            labels = np.array(labels)
            largest_cluster_idx_list = np.where(labels==largest_cluster_label)
            largest_cluster_orgn_idx_list = unclustered_idx_list[largest_cluster_idx_list]
            clustering_result[largest_cluster_orgn_idx_list] = i+1
            
            unclustered_idx_list = np.delete(unclustered_idx_list, largest_cluster_idx_list)
            unclustered_points = np.array(clustering_points.copy())[unclustered_idx_list]
            
            if unclustered_points.shape[0] == 0: 
                break
    return clustering_result

def majority_voting_masks_plants(leaf_masks_dict, clustering, mask_ids, 
                                 outlier_label=0, 
                                 outlier_color=9999, 
                                 output_img_size=(1024,1024), 
                                 min_area_th=0, 
                                 max_area_th=1024*1024) -> np.ndarray:
    
    plant_seg = np.zeros(output_img_size).astype(np.uint16)
    H, W = output_img_size
    voting_list = [[[] for w in range(W)] for h in range(H)]
    
    for clustering_label, mask_id in zip(clustering, mask_ids):
        mask = leaf_masks_dict.get(mask_id)
        if mask is None:
            continue
            
        if np.count_nonzero(mask) > max_area_th or np.count_nonzero(mask) < min_area_th:
            continue
        
        mask_ys, mask_xs = np.where(mask > 0)
        
        if max(mask_ys) >= H or max(mask_xs) >= W:
            continue
            
        if clustering_label == outlier_label:
            for y, x in zip(mask_ys, mask_xs):
                voting_list[y][x].append(outlier_color)
        else:
            for y, x in zip(mask_ys, mask_xs):
                voting_list[y][x].append(clustering_label)
    
    for h in range(H):
        for w in range(W):
            if len(voting_list[h][w]) > 0:
                plant_seg[h, w] = max(voting_list[h][w], key=voting_list[h][w].count)
    return plant_seg

def min_MHLdist_label(plant_seg, clustering_points, outlier_crd, clustering, outlier_label=0, dist_th=64):
    covmat_dict = {}
    clustering_points_arr = np.array(clustering_points)
    
    for label in np.unique(clustering):
        crds = clustering_points_arr[np.where(clustering == label)]
        if len(crds) > 1:
            covmat = np.cov(crds, rowvar=False)
            covmat_dict[label] = covmat
        else:
            covmat_dict[label] = np.eye(2) 
    
    min_dist = float('inf')
    min_dist_label = outlier_label
    
    for label in np.unique(clustering):
        if label == outlier_label:
            continue
        crnt_plant_mask = plant_seg == label
        contours = cv2.findContours(crnt_plant_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        for contour in contours:
            for pt in contour:
                try:
                    dist = distance.mahalanobis([pt[0][1], pt[0][0]], outlier_crd, covmat_dict[label])
                    if 0 < dist < min_dist and dist < dist_th:
                        min_dist = dist
                        min_dist_label = label
                except ValueError:
                    continue
    return min_dist_label

def post_process_outliers(plant_seg: np.ndarray, clustering_points: np.ndarray, 
                          clustering, outlier_label=0, dist_th=64, GENERATE_NEW_CLUSTER=False):
    
    outlier_idxes = [idx for idx, label in enumerate(clustering) if label == outlier_label]
    processed_clustering = clustering.copy()
    
    new_cluster_id = processed_clustering.max() + 1
    for outlier_idx in outlier_idxes:
        outlier_crd = clustering_points[outlier_idx]
        revised_label = min_MHLdist_label(plant_seg, clustering_points, outlier_crd, clustering, outlier_label, dist_th=dist_th)
        
        if revised_label != outlier_label:
            processed_clustering[outlier_idx] = revised_label
        elif GENERATE_NEW_CLUSTER:
            processed_clustering[outlier_idx] = new_cluster_id
            new_cluster_id += 1
            
    return processed_clustering

def plant_segmentation(bgr_img_dir, masks_dirs: list, config_path, output_dir, strip=".png", GENERATE_NEW_CLUSTER=True, VALIDATION_MODE=False):
    if not VALIDATION_MODE:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(config_path, 'r') as f:
        config_yaml = yaml.safe_load(f)
        config = config_yaml['plant_segmentation']
        
    num_steps = config['num_steps']
    min_area_th = config['min_area_th']
    
    dsize_str = config.get('output_img_size', '(1024, 1024)')
    output_img_size = str_to_tuple(dsize_str) if isinstance(dsize_str, str) else dsize_str
    
    for masks_dir in tqdm(masks_dirs, desc="Clustering Plants"):
        img_name = basename(masks_dir.rstrip('/\\'))
        img_filename = img_name + strip
        
        # use static DBSCAN thresholds
        eps = config.get('default_eps', 120.0)
        min_samples = config.get('default_min_samples', 2)
        
        # CALC LEAF KEYPOINTS via OPENCV
        clustering_points, mask_ids = get_leaf_root_opencv(masks_dir)
        
        if len(clustering_points) > 0:
            clustering = greedyDBSCANclustering(clustering_points, eps=eps, min_samples=min_samples, num_steps=num_steps)
            
            clustering = clustering + 1
            outlier_label = 0
            
            leaf_mask_dict = load_binary_masks_as_dict(masks_dir)
            plant_seg_wo_pp = majority_voting_masks_plants(leaf_mask_dict, clustering, mask_ids, outlier_label=outlier_label, output_img_size=output_img_size, min_area_th=min_area_th)
            processed_clustering = post_process_outliers(plant_seg_wo_pp, clustering_points, clustering, outlier_label, GENERATE_NEW_CLUSTER=GENERATE_NEW_CLUSTER)
            plant_seg_w_pp = majority_voting_masks_plants(leaf_mask_dict, processed_clustering, mask_ids, outlier_label=outlier_label, output_img_size=output_img_size, min_area_th=min_area_th)
        else:
            plant_seg_wo_pp = np.zeros(output_img_size).astype(np.uint16)
            plant_seg_w_pp = np.zeros(output_img_size).astype(np.uint16)
        
        if not VALIDATION_MODE:
            cv2.imwrite(os.path.join(output_dir, img_filename[:-4]+".png"), plant_seg_w_pp)
            
            # visualizer
            vis_colour = np.zeros((*plant_seg_w_pp.shape, 3), dtype=np.uint8)
            unique_plants = np.unique(plant_seg_w_pp)
            rng = np.random.RandomState(42) 
            for pid in unique_plants:
                if pid == 0: continue
                colour = rng.randint(40, 255, 3).tolist()
                vis_colour[plant_seg_w_pp == pid] = colour
            cv2.imwrite(os.path.join(output_dir, img_filename[:-4]+"_color.png"), vis_colour)

    return plant_seg_wo_pp, plant_seg_w_pp

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset name", choices={'phenobench', 'growliflower', 'sb20', 'cabbage'}, default='cabbage')
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument("--mode", type=str, help="dataset type", default='test')
    parser.add_argument("--base_path", type=str, default="./output_p")
    parser.add_argument("--strip", type=str, default=".jpg")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = setup_args()
    bgr_img_dir = args.dataset_dir
    base_path = args.base_path
    
    masks_root_dir = os.path.join(base_path, 'leaf_mask', args.dataset, args.mode)
    masks_dirs = glob.glob(os.path.join(masks_root_dir, "*/"))
    
    output_dir = os.path.join(base_path, 'plant_instance', args.dataset, args.mode)
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join('configs', args.dataset + '.yaml')
    
    plant_segmentation(bgr_img_dir, masks_dirs, config_path, output_dir=output_dir, strip=args.strip, GENERATE_NEW_CLUSTER=True)