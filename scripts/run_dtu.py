import os
import sys

if len(sys.argv) > 2:
    gpu_id=sys.argv[2]
else:
    gpu_id=0

if len(sys.argv) > 3:
    selected_idxs = [int(item) for item in sys.argv[3].split(",")]
else:
    selected_idxs = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]

print(selected_idxs, gpu_id)

scenes = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
data_base_path='./data/DTU_2dgs'
out_base_path=sys.argv[1]
eval_path='data/DTU_eval/'
out_name='test'


scenes = [scenes[i] for i in selected_idxs]


for scene in scenes:

    common_args = "-r2 --ncc_scale 0.5 --depth_weight 0.1 --sh_unc_lower_max 0.2"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/dtu_scan{scene}/{out_name} {common_args}'
    print(cmd)
    os.system(cmd)

    common_args = "--num_cluster 1 --voxel_size 0.002 --max_depth 5.0"
    if scene == 63: common_args += " --num_cluster 2"
    if scene in [63, 83, 110]: common_args += " --sdf_trunc_scale 1.5"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python mesh_extract/extract_general.py -m {out_base_path}/dtu_scan{scene}/{out_name} {common_args}'
    print(cmd)
    os.system(cmd)

    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python scripts/eval_dtu/evaluate_single_scene.py " + \
          f"--input_mesh {out_base_path}/dtu_scan{scene}/{out_name}/mesh/tsdf_fusion_post.ply " + \
          f"--scan_id {scene} --output_dir {out_base_path}/dtu_scan{scene}/{out_name}/mesh " + \
          f"--mask_dir {data_base_path} " + \
          f"--DTU {eval_path}"
    print(cmd)
    os.system(cmd)