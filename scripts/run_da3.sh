dir=$1
post_fix="_da3"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

max_points=$2       ## 50,000 for DTU, 500,000 for TnT
ransac_thresh=$3    ## 0.01 for DTU, 0.05 for TnT

export PYTHONPATH="$repo_root/multi_view_priors:${PYTHONPATH}"

# Run all scenes in the dataset directory by default
for scene_dir in $dir/*; do
    echo "Start running on '$scene_dir'"
    python "$repo_root/multi_view_priors/estimate_colmap.py" \
        --scene_dir $scene_dir \
        --post_fix $post_fix \
        --shared_camera --save_depth \
        --max_points_for_colmap $max_points

    python "$repo_root/multi_view_priors/pose_align.py" --scene1 $dir/${scene_dir##*/}/sparse$post_fix/0 --scene2 $scene_dir/sparse \
        --out $dir/${scene_dir##*/}/sparse${post_fix}_aligned/0 \
        --ransac_thresh $ransac_thresh
done
wait
