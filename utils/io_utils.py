# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import cv2
import tqdm
import torch
import numpy as np


def gather_todo_list_npy(depth_root, cameras, depth_key='depth'):

    if not os.path.exists(depth_root):
        print(f"Warning: Depth root directory not found at {depth_root}")
        return []

    npy_files = set()
    for filename in os.listdir(depth_root):
        if filename.endswith('.npy'):
            npy_files.add(filename[:-4])

    todo_indices = []
    for i, cam in enumerate(cameras):
        if hasattr(cam, depth_key):
            continue
        if os.path.basename(cam.image_path) in npy_files:
            todo_indices.append(i)
            
    return todo_indices


def load_npy_depth_to_camera(depth_root, cameras, depth_key):
    npy_paths = {}
    for filename in os.listdir(depth_root):
        if filename.endswith('.npy'):
            image_name = filename[:-4]
            npy_paths[image_name] = os.path.join(depth_root, filename)

    loaded_count = 0
    for cam in tqdm.tqdm(cameras, desc=f"Loading {depth_key} from .npy"):
        if cam.image_name in npy_paths:
            try:
                depth_np = np.load(npy_paths[cam.image_name])
                setattr(cam, depth_key, nn.Parameter(torch.tensor(depth_np, dtype=torch.float32, device="cuda").requires_grad_(True)))
                loaded_count += 1
            except Exception as e:
                print(f"Error loading {npy_paths[cam.image_name]}: {e}")
    
    print(f"Successfully loaded {loaded_count} depths to cameras.")


@torch.no_grad()
def prepare_depth(cameras, source_path):
    depth_key = "depth" 
    depth_root = os.path.join(source_path, "estimated_depths")
    todo_indices = gather_todo_list_npy(depth_root, cameras, depth_key=depth_key)
    
    if len(todo_indices) == 0:
        print(f"All cameras already have '{depth_key}' loaded or no matching .npy files found in {depth_root}.")
        return

    print(f"Found {len(todo_indices)} matching NPY files in {depth_root}.")

    print("Loading NPY depth maps to cameras.")
    load_npy_depth_to_camera(depth_root, cameras, depth_key) 