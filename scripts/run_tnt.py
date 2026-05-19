import os
import sys

if len(sys.argv) > 2:
    gpu_id=sys.argv[2]
else:
    gpu_id=0

if len(sys.argv) > 3:
    selected_idxs = [int(item) for item in sys.argv[3].split(",")]
else:
    selected_idxs = [0,1,2,3,4,5]

print(selected_idxs, gpu_id)

scenes = ['Barn', 'Courthouse', 'Truck', 'Caterpillar', 'Meetingroom', 'Ignatius']
data_devices = ['cuda', 'cpu', 'cuda', 'cuda','cuda','cuda']

scenes = [scenes[i] for i in selected_idxs]
data_devices = [data_devices[i] for i in selected_idxs]

data_base_path='data/TnT'
out_base_path=sys.argv[1]
out_name='test'

for id, scene in enumerate(scenes):
    
    common_args = f" -r2 --ncc_scale 0.5 --data_device {data_devices[id]} --densify_abs_grad_threshold 0.00015 --opacity_cull_threshold 0.05 --exposure_compensation"
    common_args += " --depth_weight 0.1 --unc_decay 0.01 --sh_ambi_lower_ratio 0.02"
    if scene in ["Barn"]: common_args += " --max_abs_split_points 100000"
    if scene in ["Caterpillar", "Courthouse"]: common_args += " --unc_weight 0.0"   # Conflict between evaluation GT to the priors
    if scene in ["Meetingroom"]: common_args += " --sh_ambi_lower_ratio 0.2 --unc_decay 1.0"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/{scene} -m {out_base_path}/{scene}/{out_name} {common_args}'
    print(cmd)
    os.system(cmd)

    common_args = f"--data_device {data_devices[id]} --num_cluster 1 --use_depth_filter"
    if scene == "Courthouse": common_args += " --max_height 1.7"
    if scene in ["Caterpillar", "Truck"]: common_args += " --sdf_trunc_scale 1.0 --num_cluster 2"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python mesh_extract/extract_tnt.py -m {out_base_path}/{scene}/{out_name} --data_device {data_devices[id]} {common_args}'
    print(cmd)
    os.system(cmd)

    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python scripts/tnt_eval/run.py --dataset-dir {data_base_path}/{scene} --traj-path {data_base_path}/{scene}/{scene}_COLMAP_SfM.log --ply-path {out_base_path}/{scene}/{out_name}/mesh/tsdf_fusion_post.ply --out-dir {out_base_path}/{scene}/{out_name}/mesh'
    print(cmd)
    os.system(cmd)