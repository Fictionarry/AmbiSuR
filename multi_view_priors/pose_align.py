#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Align two COLMAP sparse reconstructions (scene1 → scene2)
and save a fully aligned sparse model.
(V4: Implements RANSAC for robust alignment against outliers)
"""

import os
import argparse
import numpy as np
from colmap.scripts.python.read_write_model import read_model, write_model, rotmat2qvec, qvec2rotmat
import json

# --- RANSAC ---
RANSAC_MAX_ITER = 20000
RANSAC_THRESHOLD = 0.05
RANSAC_MIN_INLIERS = 3 

def image_to_matrix(image):
    R = qvec2rotmat(image.qvec)
    t = image.tvec.reshape(3, 1)
    W2C = np.eye(4)
    W2C[:3, :3] = R
    W2C[:3, 3] = t[:, 0]
    return W2C

def compute_umeyama_transform(src, dst):
    mu1, mu2 = np.mean(src, axis=0), np.mean(dst, axis=0)
    X1, X2 = src - mu1, dst - mu2
    N = len(src)
    H = X1.T @ X2 / N
    U, S, Vt = np.linalg.svd(H)
    
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
        
    var1 = np.sum(X1**2) / N
    if var1 < 1e-9:
        scale = 1.0
    else:
        scale = np.sum(S) / var1

    t = mu2 - scale * R @ mu1

    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t
    return T, scale

def align_scenes(images1, images2, max_iter=RANSAC_MAX_ITER, threshold=RANSAC_THRESHOLD):
    name_to_img1 = {img.name: img for img in images1.values()}
    name_to_img2 = {img.name: img for img in images2.values()}
    
    common_names = sorted(set(name_to_img1.keys()) & set(name_to_img2.keys()))
    N = len(common_names)
    
    print(f"Found {N} common images for alignment. Applying RANSAC...")
    if N < RANSAC_MIN_INLIERS:
        raise ValueError(f"Need at least {RANSAC_MIN_INLIERS} common images to compute alignment.")

    pts1_all, pts2_all = [], []
    for name in common_names:
        c2w1 = np.linalg.inv(image_to_matrix(name_to_img1[name]))
        c2w2 = np.linalg.inv(image_to_matrix(name_to_img2[name]))
        pts1_all.append(c2w1[:3, 3])
        pts2_all.append(c2w2[:3, 3])

    pts1_all = np.stack(pts1_all)
    pts2_all = np.stack(pts2_all)
    
    best_T = np.eye(4)
    best_scale = 1.0
    best_inlier_count = 0
    
    for i in range(max_iter):
        sample_indices = np.random.choice(N, RANSAC_MIN_INLIERS, replace=False)
        pts1_sample = pts1_all[sample_indices]
        pts2_sample = pts2_all[sample_indices]
        
        try:
            T_test, _ = compute_umeyama_transform(pts1_sample, pts2_sample)
        except np.linalg.LinAlgError:
            continue

        pts1_h = np.hstack([pts1_all, np.ones((N, 1))])
        pts1_transformed = (T_test @ pts1_h.T).T[:, :3]
        
        errors = np.linalg.norm(pts1_transformed - pts2_all, axis=1)
        inliers = errors < threshold
        inlier_count = np.sum(inliers)
        
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            
            pts1_inliers = pts1_all[inliers]
            pts2_inliers = pts2_all[inliers]
            
            if len(pts1_inliers) >= RANSAC_MIN_INLIERS:
                best_T, best_scale = compute_umeyama_transform(pts1_inliers, pts2_inliers)
                # print(f"Iter {i}: New best inliers: {best_inlier_count}/{N}. Scale: {best_scale:.4f}")
            
    if best_inlier_count < RANSAC_MIN_INLIERS:
         print(f"RANSAC Warning: Only found {best_inlier_count} inliers (min required: {RANSAC_MIN_INLIERS}). T might be unreliable.")

    print(f"RANSAC Final Result: {best_inlier_count}/{N} inliers ({best_inlier_count/N*100:.1f}%).")
    print(f"Computed scale factor: {best_scale:.4f}")
    
    return best_T, best_scale


def apply_alignment(images, points3D, T):

    U, S, Vt = np.linalg.svd(T[:3, :3])
    R_sim = U @ Vt
    if np.linalg.det(R_sim) < 0:
        Vt[-1, :] *= -1
        R_sim = U @ Vt
    
    scale_sim = S.mean()
    t_sim = T[:3, 3]
    
    
    aligned_images = {}
    for image_id, img in images.items():
        w2c_old = image_to_matrix(img)
        c2w_old = np.linalg.inv(w2c_old)
        R_c2w_old = c2w_old[:3, :3]
        t_c2w_old = c2w_old[:3, 3]

        R_c2w_new = R_sim @ R_c2w_old
        t_c2w_new = scale_sim * (R_sim @ t_c2w_old) + t_sim

        c2w_new = np.eye(4)
        c2w_new[:3, :3] = R_c2w_new
        c2w_new[:3, 3] = t_c2w_new
        
        w2c_new = np.linalg.inv(c2w_new)
        R_new = w2c_new[:3, :3]
        t_new = w2c_new[:3, 3]
        q_new = rotmat2qvec(R_new)

        new_img = type(img)(
            id=img.id,
            qvec=q_new,
            tvec=t_new,
            camera_id=img.camera_id,
            name=img.name,
            xys=img.xys,
            point3D_ids=img.point3D_ids,
        )
        aligned_images[image_id] = new_img

    aligned_points = {}
    for pid, p in points3D.items():
        xyz_h = np.concatenate([p.xyz, [1]])
        xyz_new = (T @ xyz_h)[:3]

        new_p = type(p)(
            id=p.id,
            xyz=xyz_new,
            rgb=p.rgb,
            error=p.error,
            image_ids=p.image_ids,
            point2D_idxs=p.point2D_idxs,
        )
        aligned_points[pid] = new_p

    return aligned_images, aligned_points

def find_model_path(base_path):
    path_bin = os.path.join(base_path, '0')
    if os.path.exists(os.path.join(path_bin, 'images.bin')):
        return path_bin, '.bin'
        
    path_txt = os.path.join(base_path, '1')
    if os.path.exists(os.path.join(path_txt, 'images.txt')):
        return path_txt, '.txt'

    if os.path.exists(os.path.join(base_path, 'images.bin')):
        return base_path, '.bin'
    if os.path.exists(os.path.join(base_path, 'images.txt')):
        return base_path, '.txt'
        
    raise FileNotFoundError(f"Cannot find COLMAP model (images.bin/txt) in {base_path} or its subdirs '0'/'1'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene1", required=True, help="Path to COLMAP model dir for scene 1 (source)")
    parser.add_argument("--scene2", required=True, help="Path to COLMAP model dir for scene 2 (target)")
    parser.add_argument("--out", required=True, help="Output path for aligned scene 1 model")
    parser.add_argument("--ransac_iter", type=int, default=RANSAC_MAX_ITER, help=f"Max RANSAC iterations (default: {RANSAC_MAX_ITER}).")
    parser.add_argument("--ransac_thresh", type=float, default=RANSAC_THRESHOLD, help=f"RANSAC inlier threshold in scene units (default: {RANSAC_THRESHOLD}).")
    args = parser.parse_args()

    try:
        path1, ext1 = find_model_path(args.scene1)
        path2, ext2 = find_model_path(args.scene2)
    except FileNotFoundError as e:
        print(e)
        return

    if ext1 != ext2:
        print(f"Warning: Model formats differ (scene1: {ext1}, scene2: {ext2}). Reading both.")

    print(f"Reading scene 1 from {path1} (format {ext1})")
    cams1, imgs1, pts1 = read_model(path1, ext=ext1)
    
    print(f"Reading scene 2 from {path2} (format {ext2})")
    cams2, imgs2, pts2 = read_model(path2, ext=ext2)

    print("Computing alignment...")
    T, scale = align_scenes(imgs1, imgs2, max_iter=args.ransac_iter, threshold=args.ransac_thresh)
    print("Transform (T_scene2_from_scene1):\n", T)
    print(f"Scale: {scale:.4f}")

    print("Applying alignment to scene 1...")
    imgs_aligned, pts_aligned = apply_alignment(imgs1, pts1, T)

    os.makedirs(args.out, exist_ok=True)
    print(f"Writing aligned model to {args.out} (format {ext1})")
    write_model(cams1, imgs_aligned, pts_aligned, path=args.out, ext=ext1)

    transform_data = {
        "scale": float(scale),
        "T_matrix_scene2_from_scene1": T.tolist()
    }
    with open(os.path.join(args.out, "trans.json"), 'w') as f:
        json.dump(transform_data, f, indent=4)

    print(f"✅ Done! Aligned model saved to {args.out}")

if __name__ == "__main__":
    main()