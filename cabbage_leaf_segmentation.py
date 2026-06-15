import os
import glob
import yaml
import argparse
import csv
import cv2
import numpy as np
from tqdm import tqdm

from utils.path import basename
from utils.fileio import str_to_tuple

def area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))

def overlap_ratio(mask1: np.ndarray, mask2: np.ndarray) -> float:
    m1_bin = mask1 > 0
    m2_bin = mask2 > 0
    intersection = np.logical_and(m1_bin, m2_bin).sum()
    smaller_area = min(area(m1_bin), area(m2_bin))

    if smaller_area > 0:
        return intersection / smaller_area
    else:
        return 0.0

def merge_leaf_masks(masks_dir: str, min_area_th: float=0, max_area_th: float=np.inf, th_ratio=0.9, output_img_size=(1024, 1024)):
    
    # load metadata
    metadata_filename = 'metadata.csv'
    csv_path = os.path.join(masks_dir, metadata_filename)
    
    # Check if directory actually has masks
    if not os.path.exists(csv_path):
        return np.zeros(output_img_size, dtype=np.uint16)
        
    with open(csv_path) as csvfile:
        reader = csv.DictReader(csvfile)
        metadata = [data for data in reader]
    metadata = sorted(metadata, key=lambda x: int(x['area']), reverse=True) # sort by area in descending order
    
    # load mask images
    masks = []
    for mask_data in metadata:
        # load mask
        try:
            mask_filename = mask_data['id'] + '.png'
        except KeyError:
            mask_filename = mask_data['filename']
            
        mask_path = os.path.join(masks_dir, mask_filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            masks.append(mask)
    
    # integration (Custom Lightweight NMS)
    masks_not_duplicate = []
    for mask in masks:
        # checkif already kept (larger) masks
        if not any(overlap_ratio(mask, kept_mask) > th_ratio for kept_mask in masks_not_duplicate):
            masks_not_duplicate.append(mask)
    
    # make output
    output_img = np.zeros(output_img_size).astype(np.uint16) # init output img
    
    # resize output canvas dynamically to match the first mask if standard size is incorrect
    if len(masks_not_duplicate) > 0 and masks_not_duplicate[0].shape != output_img_size:
        output_img = np.zeros(masks_not_duplicate[0].shape).astype(np.uint16)
        
    for i, mask in enumerate(masks_not_duplicate):
        if area(mask) > max_area_th or area(mask) < min_area_th:
            continue
        output_img[np.where(mask != 0)] = i + 1
    
    return output_img

def make_leaf_seg(masks_dirs: list, output_dir: str, config, VALIDATION_MODE=False):
    
    if not VALIDATION_MODE:
        os.makedirs(output_dir, exist_ok=True)
    else:
        outputs = []
    
    # load parameters
    with open(config) as f:
        config_yaml = yaml.safe_load(f)
        config_params = config_yaml['leaf_segmentation']
        
    min_area_th = config_params.get('min_area_th', 500)
    max_area_th = float(config_params.get('max_area_th', np.inf))
    th_ratio = config_params.get('th_ratio', 0.9)
    
    # handle string tuples dynamically for yaml
    dsize_str = config_params.get('output_img_size', '(1024, 1024)')
    output_img_size = str_to_tuple(dsize_str) if isinstance(dsize_str, str) else dsize_str
    
    # integration
    for masks_dir in tqdm(masks_dirs, desc="Merging Leaf Masks"):
        leaf_segmentation_output = merge_leaf_masks(
            masks_dir, 
            min_area_th=min_area_th, 
            max_area_th=max_area_th, 
            th_ratio=th_ratio, 
            output_img_size=output_img_size
        )

        img_name = basename(masks_dir.rstrip('/\\')) + '.png'
        
        # output
        if not VALIDATION_MODE:
            cv2.imwrite(os.path.join(output_dir, img_name), leaf_segmentation_output)
        else:
            outputs.append(leaf_segmentation_output)
    
    if VALIDATION_MODE:
        return outputs
    else:
        return
    
def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset name", choices={'phenobench', 'growliflower', 'sb20', 'cabbage'})
    parser.add_argument("--mode", type=str, help="dataset type", default="test")
    parser.add_argument("--strip", type=str, default="*.png")
    parser.add_argument("--base_path", type=str, default="./output_p")
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = setup_args()
    base_path = args.base_path
    
    masks_root_dir = os.path.join(base_path, 'leaf_mask', args.dataset, args.mode)
    masks_dirs = glob.glob(os.path.join(masks_root_dir, "*/"))
    
    output_dir = os.path.join(base_path, 'leaf_instance', args.dataset, args.mode)
    os.makedirs(output_dir, exist_ok=True)
    
    config_path = os.path.join('configs', args.dataset + '.yaml')
    
    make_leaf_seg(masks_dirs, output_dir, config_path)