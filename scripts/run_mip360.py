import os
import sys

if len(sys.argv) > 2:
    gpu_id=sys.argv[2]
else:
    gpu_id=0

if len(sys.argv) > 3:
    selected_idxs = [int(item) for item in sys.argv[3].split(",")]
else:
    selected_idxs = range(0, 9)

print(selected_idxs, gpu_id)

scenes = ['bicycle', 'bonsai', 'counter', 'flowers', 'garden', 'kitchen', 'room', 'stump', 'treehill']
factors = ['4', '2', '2', '4', '4', '2', '2', '4', '4']
data_devices = ['cpu', 'cuda', 'cuda', 'cuda', 'cuda', 'cuda', 'cuda', 'cuda', 'cuda']

scenes = [scenes[i] for i in selected_idxs]
factors = [factors[i] for i in selected_idxs]
data_devices = [data_devices[i] for i in selected_idxs]

data_base_path='data/360_v2'
out_base_path=sys.argv[1]
out_name='test'

for id, scene in enumerate(scenes):

    common_args = f"-r{factors[id]} --data_device {data_devices[id]} --densify_abs_grad_threshold 0.0002 --eval --trunc_sigma 4.0"
    common_args += " --depth_weight 0.1 --unc_decay 0.01 --sh_ambi_lower_ratio 0.02 --use_mono --ray_color_lambda 1e-7"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/{scene} -m {out_base_path}/{scene}/{out_name} {common_args}'
    print(cmd)
    os.system(cmd)

    common_args = f"--skip_train"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python mesh_extract/extract_general.py -m {out_base_path}/{scene}/{out_name} {common_args}' 
    print(cmd)
    os.system(cmd)
    
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python metrics.py -m {out_base_path}/{scene}/{out_name}'
    print(cmd)
    os.system(cmd)
    
    common_args = f""
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python mesh_extract/extract_adaptive.py -m {out_base_path}/{scene}/{out_name} {common_args}' 
    print(cmd)
    os.system(cmd)
    
    common_args = f""
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python render_mesh.py -m {out_base_path}/{scene}/{out_name} {common_args}' 
    print(cmd)
    os.system(cmd)
