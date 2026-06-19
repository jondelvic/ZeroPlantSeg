import os
import glob
import argparse
import yaml
from tqdm import tqdm

import cv2
import numpy as np
import torch
from PIL import Image
import open_clip
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from utils.path import filename_wo_ext
from utils.fileio import str_to_tuple
from sliding_window.sliding_window import sliding_window, save_masks, save_masks_not_on_boundary

class SAMVisualizationDemo:
    def __init__(self, granularity, sam_weights, class_names, clip_score_thresh=0.7):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # scale SAM's points_per_side rather than OVSeg's original use
        points_per_side = max(8, int(32 * granularity)) if granularity else 32
        sam = sam_model_registry["vit_b"](checkpoint=sam_weights).to(device=self.device)
        self.mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=points_per_side, pred_iou_thresh=0.88, crop_n_layers=1)

        # init OpenCLIP
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        self.clip_model.to(self.device)
        self.clip_model.eval()
        self.clip_tokenizer = open_clip.tokenize
        self.clip_score_thresh = clip_score_thresh

        with torch.no_grad():
            text_tokens = self.clip_tokenizer(class_names).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        self.text_features = text_features

    def run_on_image(self, img_bgr, class_names):
        scale = 1024 / min(img_bgr.shape[:2]) #forgot to readd this hahaha OOMpsies
        if scale < 1.0:
            img_bgr = cv2.resize(img_bgr, (int(img_bgr.shape[1] * scale), int(img_bgr.shape[0] * scale)))

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        raw_masks = self.mask_generator.generate(rgb)

        ins_segs = [[] for _ in class_names]

        for m in raw_masks:
            mask_bin = m["segmentation"].astype(np.uint8)
            area = int(np.count_nonzero(mask_bin))

            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

            margin = max(5, int(min(w, h) * 0.2))
            y1, y2 = max(y - margin, 0), min(y + h + margin, rgb.shape[0])
            x1, x2 = max(x - margin, 0), min(x + w + margin, rgb.shape[1])

            # mask region will be pasted in a neutral gray bg (since CLIP was trained on natural images) instead of black
            crop_rgb = rgb[y1:y2, x1:x2]
            crop_mask = (mask_bin[y1:y2, x1:x2] > 0)[..., np.newaxis]
            gray_bg = np.full_like(crop_rgb, 127)
            crop_masked = np.where(crop_mask, crop_rgb, gray_bg)

            img_tensor = self.clip_preprocess(Image.fromarray(crop_masked)).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.clip_model.encode_image(img_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                # text_features cached at init since class_names is fixed
                similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)

            # assumes single-image batch (batch size 1)
            best_cls_idx = similarity.argmax().item()
            print(f"area={area:6d} | class={class_names[best_cls_idx]:20s} | score={similarity[0, best_cls_idx].item():.3f}")
            if similarity[0, best_cls_idx].item() > self.clip_score_thresh:
                ins_segs[best_cls_idx].append({
                    "segmentation": mask_bin,
                    "area": area,
                    "score": similarity[0, best_cls_idx].item(),
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "predicted_iou": m["predicted_iou"],
                    "stability_score": m["stability_score"]
                })

        return None, None, ins_segs

def inference(img_paths: list, output_dir: str, config_path: str):

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        config = config['leaf_mask_collection']
    class_names = config['class_names']
    class_names = class_names.split(',')
    trg_class_names = config['trg_class_names']
    trg_class_names = trg_class_names.split(',')
    granularity = config['granularity']
    min_mask_area = config['min_mask_area']
    clip_score_thresh = config.get('clip_score_thresh', 0.7)


    demo = SAMVisualizationDemo(granularity, './checkpoints/sam_vit_b_01ec64.pth', class_names, clip_score_thresh)
    for img_path in tqdm(img_paths):

        img = cv2.imread(img_path)

        # inference on raw img
        leaf_segs = []
        _, _, ins_segs = demo.run_on_image(img, class_names)
        for cls_idx, ins_seg in enumerate(ins_segs):
            if class_names[cls_idx] in trg_class_names:
                for item in ins_seg:
                    item['win_id'] = 999 # id for not cropped (raw) img
                leaf_segs.extend(ins_seg)

        if config['use_sliding_window']:
            crop_imgs = sliding_window(img, str_to_tuple(config['dsize']))

            # inference on each cropped img
            for crop_id, crop_img in enumerate(crop_imgs):

                _, _, ins_segs = demo.run_on_image(crop_img, class_names)

                # extend leaf_segs with masks of each cropped img
                for cls_idx, ins_seg in enumerate(ins_segs):
                    if class_names[cls_idx] in trg_class_names:
                        for item in ins_seg:
                            item['win_id'] = crop_id
                        leaf_segs.extend(ins_seg)

            # save masks not on boundary
            img_name = filename_wo_ext(img_path)
            if len(trg_class_names) == 1:
                save_folder = os.path.join(output_dir, img_name)
                os.makedirs(save_folder, exist_ok=True)
                save_masks_not_on_boundary(leaf_segs, save_folder, win_size=(512,512), min_area=min_mask_area)
            else:
                for cls_idx, trg_class_name in enumerate(trg_class_names):
                    save_folder = os.path.join(output_dir, img_name, trg_class_name)
                    os.makedirs(save_folder, exist_ok=True)
                    save_masks_not_on_boundary(leaf_segs[cls_idx], save_folder, win_size=(512,512), min_area=min_mask_area)
        else:
            # save all masks
            img_name = filename_wo_ext(img_path)
            save_folder = os.path.join(output_dir, img_name)
            os.makedirs(save_folder, exist_ok=True)
            save_masks(leaf_segs, save_folder, min_mask_area)

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset name", choices={'phenobench', 'growliflower', 'sb20', 'cabbage'})
    parser.add_argument('--dataset_dir',type=str)
    parser.add_argument("--mode", type=str, help="dataset type", )
    parser.add_argument("--strip",type=str,default="*.png")
    parser.add_argument("--base_path",type=str, default="./output_p")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = setup_args()

    #img_dir = os.path.join('/datasets', args.dataset, args.mode, 'images')
    img_dir = args.dataset_dir
    img_paths = glob.glob(os.path.join(img_dir, args.strip))
    output_dir = os.path.join(args.base_path,'leaf_mask', args.dataset, args.mode)
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join('configs', args.dataset + '.yaml')

    inference(img_paths, output_dir, config_path)
