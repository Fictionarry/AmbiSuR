# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
import glob
import os
import torch
import torch.nn.functional as F

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import argparse
import trimesh
import vggt.utils.colmap as colmap_utils
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

from depth_anything_3.api import DepthAnything3
from vggt.utils.load_fn import load_and_preprocess_images_ratio
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track


torch._dynamo.config.accumulated_cache_size_limit = 512

def run_DA3(images, device, dtype):
    device = torch.device("cuda")
    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
    model = model.to(device=device)

    chunk_size = 450
    
    extrinsic, intrinsic, depth_map, depth_conf = [], [], [], []
    
    for i in range((len(images) - 1) // chunk_size + 1):
        images_sub = [images[0]] + images[chunk_size*i:chunk_size*(i+1)]
        with torch.no_grad():
            prediction = model.inference(
                images_sub,
                # export_format="npz-glb",
                # export_dir="output-test",
            )
            extrinsic += prediction.extrinsics[1:].tolist()
            intrinsic += prediction.intrinsics[1:].tolist()
            depth_map += prediction.depth[1:].tolist()
            depth_conf += prediction.conf[1:].tolist()
    
    return np.asarray(extrinsic), np.asarray(intrinsic), np.asarray(depth_map), np.asarray(depth_conf)

def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Demo")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images")
    parser.add_argument("--post_fix", type=str, default="_da3", help="Post fix for the output sparse folder")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--save_depth", action="store_true", default=False, help="If save depth")
    parser.add_argument("--total_frame_num", type=int, default=None, help="Number of frames to reconstruct")
    parser.add_argument("--max_points_for_colmap", type=int, default=500000, help="Maximum number for colmap point cloud")
    parser.add_argument("--conf_percent", type=int, default=20, help="Percentile of filtered depth point cloud.")
    parser.add_argument("--shared_camera", action="store_true", default=False, help="Use shared camera for all images")
    return parser.parse_args()

def demo_fn(args):
    # Print configuration
    print("Arguments:", vars(args))

    target_scene_dir = args.scene_dir
    print(f"Outputting directly to original scene dir: {target_scene_dir}")

    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    print(f"Setting seed as: {args.seed}")

    # Set device and dtype
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Get image paths and preprocess them
    image_dir = os.path.join(args.scene_dir, "images")
    if args.total_frame_num is None:
        args.total_frame_num = len(os.listdir(image_dir))

    if os.path.exists(os.path.join(args.scene_dir, "sparse/0/images.bin")):
        print("Using order of ground truth images from COLMAP sparse reconstruction")
        images_gt = colmap_utils.read_images_binary(os.path.join(args.scene_dir, "sparse/0/images.bin"))
        assert args.total_frame_num <= len(images_gt), f"Requested total_frame_num {args.total_frame_num} exceeds available images {len(images_gt)}"
        
        images_gt = dict(list(images_gt.items())[:args.total_frame_num])
        images_gt_keys = list(images_gt.keys())

        random.shuffle(images_gt_keys)
        images_gt_updated = {id: images_gt[id] for id in list(images_gt_keys)}
        image_path_list = [os.path.join(image_dir, images_gt_updated[id].name) for id in images_gt_updated.keys()]

        inverse_idx = [images_gt_keys.index(key) for key in sorted(list(images_gt.keys()))]
    else:
        image_path_list = sorted(glob.glob(os.path.join(image_dir, "*")))[:args.total_frame_num]
        if not image_path_list:
            raise ValueError(f"No images found in {image_dir}")
        inverse_idx = list(range(len(image_path_list)))

    base_image_path_list = [os.path.basename(path) for path in image_path_list]
    base_image_path_list_inv = [base_image_path_list[i] for i in inverse_idx]

    # Load images and original coordinates
    # Run DA3 with 504
    img_load_resolution = 504

    images, original_coords = load_and_preprocess_images_ratio(image_path_list, img_load_resolution)
    original_coords = original_coords.to(device)
    print(f"Loaded {len(images)} images from {image_dir}")

    torch.cuda.reset_peak_memory_stats()
    start_time = datetime.now()

    extrinsic, intrinsic, depth_map, depth_conf = run_DA3(image_path_list, device, dtype)
    
    images = images.to(device)
        
    end_time = datetime.now()
    peak_mem_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0

    conf_thres_value = np.percentile(depth_conf, args.conf_percent)
    print(f"Using confidence threshold: {conf_thres_value}")
    shared_camera = args.shared_camera
    camera_type = "PINHOLE"  # in colmap result saving, we only support PINHOLE camera

    c = 2.5  # scale factor for better reconstruction, hard-coded here
    extrinsic[:, :3, 3] *= c
    depth_map *= c

    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    if args.save_depth:
        # save depth_map and depth_conf as .npy files
        target_depth_dir = os.path.join(target_scene_dir, "estimated_depths")
        target_conf_dir = os.path.join(target_scene_dir, "estimated_confs")
        os.makedirs(target_depth_dir, exist_ok=True)
        os.makedirs(target_conf_dir, exist_ok=True)

        for idx, image_path in tqdm(enumerate(image_path_list), desc="Saving depth maps and confidences"):
            depth_map_path = os.path.join(target_depth_dir, f"{os.path.basename(image_path)}.npy")
            depth_conf_path = os.path.join(target_conf_dir, f"{os.path.basename(image_path)}.npy")
            np.save(depth_map_path, depth_map[idx].squeeze())
            np.save(depth_conf_path, depth_conf[idx].squeeze())

            inverse_depth_map = 1 / (depth_map[idx] + 1e-8)  # Avoid division by zero
            normalized_inverse_depth_map = (inverse_depth_map - inverse_depth_map.min()) / (inverse_depth_map.max() - inverse_depth_map.min())
            import torchvision
            torchvision.utils.save_image(torch.asarray(normalized_inverse_depth_map), os.path.join(target_depth_dir, f"{os.path.basename(image_path)}.jpg"))

    image_size = np.array([depth_map.shape[1], depth_map.shape[2]])
    num_frames, height, width, _ = points_3d.shape

    points_rgb = F.interpolate(
        images, size=(depth_map.shape[1], depth_map.shape[2]), mode="bilinear", align_corners=False
    )
    points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
    points_rgb = points_rgb.transpose(0, 2, 3, 1)

    # (S, H, W, 3), with x, y coordinates and frame indices
    points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

    conf_mask = depth_conf >= conf_thres_value
    # at most writing args.max_points_for_colmap 3d points to colmap reconstruction object
    conf_mask = randomly_limit_trues(conf_mask, args.max_points_for_colmap)

    points_3d = points_3d[conf_mask]
    points_xyf = points_xyf[conf_mask]
    points_rgb = points_rgb[conf_mask]

    print("Converting to COLMAP format")
    reconstruction = batch_np_matrix_to_pycolmap_wo_track(
        points_3d,
        points_xyf,
        points_rgb,
        extrinsic[inverse_idx],
        intrinsic[inverse_idx],
        image_size,
        shared_camera=shared_camera,
        camera_type=camera_type,
    )

    reconstruction_resolution = (depth_map.shape[2], depth_map.shape[1])

    reconstruction = colmap_utils.rename_colmap_recons_and_rescale_camera(
        reconstruction,
        base_image_path_list_inv,
        original_coords.cpu().numpy()[inverse_idx],
        img_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
    )

    sparse_out_dir_name = f"sparse{args.post_fix}"
    print(f"Saving reconstruction to {target_scene_dir}/{sparse_out_dir_name}/0")
    sparse_reconstruction_dir = os.path.join(target_scene_dir, f"{sparse_out_dir_name}/0")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)

    # Save point cloud for fast visualization
    trimesh.PointCloud(points_3d, colors=points_rgb).export(os.path.join(target_scene_dir, f"{sparse_out_dir_name}/points.ply"))

    return True

if __name__ == "__main__":
    args = parse_args()
    demo_fn(args)