# A tool to quickly count the mean metrics of one dir
# usage:
# $ python metrics_count.py output/ 6000

import os
import json

import numpy as np 
import sys

dataset_path = sys.argv[1]
model_id = "ours_30000"


ssims_gs = []
psnrs = []
lpipss = []
avgs = []


indoor_scenes = ['bonsai', 'counter', 'kitchen', 'room']
outdoor_scenes = ['bicycle', 'garden', 'stump', 'treehill', 'flowers']

def psnr_to_mse(psnr):
  """Compute MSE given a PSNR (we assume the maximum pixel value is 1)."""
  return np.exp(-0.1 * np.log(10.) * psnr)

def compute_avg_error(psnr, ssim, lpips):
  """The 'average' error used in the paper."""
  mse = psnr_to_mse(psnr)
  dssim = np.sqrt(1 - ssim)
  return np.exp(np.mean(np.log(np.array([mse, dssim, lpips]))))


for fname in indoor_scenes + outdoor_scenes:
    if not os.path.isdir(os.path.join(dataset_path, fname)) or not os.path.exists(os.path.join(dataset_path, fname, 'test', 'results.json')): 
      print(f"{fname} no results") 
      continue
    with open(os.path.join(dataset_path, fname, 'test', 'results.json')) as f:
        result=json.load(f)
    if model_id not in result: 
      print(f"{fname} no {model_id}")
      continue
    ssims_gs.append(result[model_id]["SSIM"])
    psnrs.append(result[model_id]["PSNR"])
    lpipss.append(result[model_id]["LPIPS"])
    # avgs.append(compute_avg_error(psnrs[-1], ssims_sk[-1], lpipss[-1]))
    print(fname, result[model_id]["PSNR"], result[model_id]["SSIM"], result[model_id]["LPIPS"])

# print(np.mean(psnrs), np.mean(lpipss), np.mean(ssims_sk), np.mean(ssims_gs), np.mean(avgs))
print("==" * 30)
print("Indoor", np.mean(psnrs[:4]), np.mean(ssims_gs[:4]), np.mean(lpipss[:4]))
print("Outdoor", np.mean(psnrs[4:]), np.mean(ssims_gs[4:]), np.mean(lpipss[4:]))
print("Overall", np.mean(psnrs), np.mean(ssims_gs), np.mean(lpipss))