# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import cv2
import tqdm
import torch
import numpy as np
import imageio.v2 as iio
from PIL import Image


def depth_path(depth_root, cam):
    return os.path.join(depth_root, f"{cam.image_name}.png")

def codebook_path(depth_root, cam):
    return os.path.join(depth_root, f"{cam.image_name}.npy")

def gather_todo_list(depth_root, cameras, force_rerun=False):
    # Gather list of camera to estimate depth
    todo_indices = []
    for i, cam in enumerate(cameras):
        if not os.path.exists(depth_path(depth_root, cam)) or force_rerun:
            todo_indices.append(i)
    return todo_indices

def load_depth_to_camera(depth_root, cameras, depth_name):
    for cam in tqdm.tqdm(cameras):
        depth_np = iio.imread(depth_path(depth_root, cam))
        codebook = np.load(codebook_path(depth_root, cam))
        setattr(cam, depth_name, torch.tensor(codebook[depth_np]))

def save_quantize_depth(depth_root, cam, depth):
    # Quantize depth map to 16 bit
    codebook = depth.quantile(torch.linspace(0, 1, 65536).cuda(), interpolation='nearest')
    depth_idx = torch.searchsorted(codebook, depth, side='right').clamp_max_(65535)
    depth_idx[(depth - codebook[depth_idx-1]).abs() < (depth - codebook[depth_idx]).abs()] -= 1
    assert depth_idx.max() <= 65535
    assert depth_idx.min() >= 0

    # Save result
    depth_np = depth_idx.cpu().numpy().astype(np.uint16)
    iio.imwrite(depth_path(depth_root, cam), depth_np)
    np.save(codebook_path(depth_root, cam), codebook.cpu().numpy().astype(np.float32))

def resize_maxres_divisible(im, len, divisible):
    max_res = max(im.shape[-2:])
    target_size = (
        divisible * round(len * im.shape[-2] / max_res / divisible),
        divisible * round(len * im.shape[-1] / max_res / divisible))
    im = torch.nn.functional.interpolate(im, size=target_size, mode='bilinear', antialias=True)
    return im


@torch.no_grad()
def prepare_depthanythingv2(cameras, source_path, force_rerun=False):

    depth_root = os.path.join(source_path, "mono_priors", "depthanythingv2")
    os.makedirs(depth_root, exist_ok=True)

    todo_indices = gather_todo_list(depth_root, cameras, force_rerun=force_rerun)
    
    if len(todo_indices):
        print(f"Infer depth for {len(todo_indices)} images. Saved to {depth_root}.")

        # Load model
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
        model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf").cuda()

    for i in tqdm.tqdm(todo_indices):
        cam = cameras[i]

        # Inference depth
        image, _ = cam.get_image()
        inputs = image_processor(images=image, return_tensors="pt", do_rescale=False)
        inputs['pixel_values'] = inputs['pixel_values'].cuda()
        outputs = model(**inputs)
        depth = outputs['predicted_depth'].squeeze()

        # Save result
        save_quantize_depth(depth_root, cam, depth)

    # Load the estimated depth
    print("Load the estimated depths to cameras.")
    load_depth_to_camera(depth_root, cameras, 'depthanythingv2')