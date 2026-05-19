#
# Adapted Render Mesh Script
# Adapts Script 1's Open3D rendering logic to Script 2's Data/API structure
#

import os
import numpy as np
from tqdm import tqdm
import open3d as o3d
from argparse import ArgumentParser

# --- Imports from Script 2's environment ---
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from gaussian_renderer import GaussianModel

if __name__ == "__main__":
    # 1. Argument Parsing (Aligned with Script 2)
    parser = ArgumentParser(description="Mesh rendering script using Gaussian Splatting dataset API.")
    
    # Load standard Gaussian Splatting arguments
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    
    # Add arguments from Script 1 that are still relevant
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    
    parser.add_argument("--mesh_path", default="", type=str, help="Path to the mesh .ply file. If empty, tries to find it in default location.")
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--suffix", default="", type=str)

    args = get_combined_args(parser)
    print("Rendering with model: " + args.model_path)

    print(args)
    
    model = model.extract(args)
    print(model)

    # 2. Load Data/Scene (Replacing DataPack with Scene)
    # Initialize GaussianModel just to satisfy Scene constructor, we won't strictly use the gaussians for mesh rendering
    gaussians = GaussianModel(model.sh_degree)
    scene = Scene(model, gaussians, load_iteration=args.iteration, shuffle=False)

    # 3. Locate and Load Mesh
    # Script 2 typically saves meshes to 'model_path/mesh/tsdf_fusion_post.ply'
    default_mesh_path = os.path.join(args.model_path, "mesh", "tsdf_fusion_post.ply")
    mesh_path = args.mesh_path if args.mesh_path else default_mesh_path

    if not os.path.exists(mesh_path):
        # Fallback to pre-post-process mesh if needed
        fallback_path = os.path.join(args.model_path, "mesh", "tsdf_fusion.ply")
        if os.path.exists(fallback_path) and args.mesh_path == "":
             mesh_path = fallback_path
        else:
            print(f"Error: Could not find mesh at {mesh_path}")
            exit(1)

    print(f"Loading mesh from {mesh_path}...")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    
    # 4. Mesh Pre-processing (Logic from Script 1)
    # Clears existing colors/materials and applies coordinate fix
    mesh.vertex_colors = o3d.utility.Vector3dVector() 
    mesh.triangle_material_ids = o3d.utility.IntVector() 
    
    vertices = np.asarray(mesh.vertices)
    # Script 1 flips Y and Z for Open3D compatibility
    vertices[:, 1] *= -1  
    vertices[:, 2] *= -1 
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    
    mesh.compute_vertex_normals()

    # 5. Renderer Setup (Logic from Script 1)
    
    # Determine resolution from the first available camera
    views = []
    if not args.skip_train:
        views += scene.getTrainCameras()
    if not args.skip_test:
        views += scene.getTestCameras()
        
    if len(views) == 0:
        print("No cameras found.")
        exit(0)

    # Assuming all cameras have roughly same resolution for renderer init, 
    # or we use the first one to init the OffscreenRenderer.
    w, h = views[0].image_width, views[0].image_height

    # --- Simple Material Setup ---
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.base_color = [0.8, 0.8, 0.8, 1.0] 
    material.base_roughness = 0.5
    material.base_metallic = 0.0
    material.base_reflectance = 0.2 

    # --- Simple Renderer ---
    renderer_simple = o3d.visualization.rendering.OffscreenRenderer(w, h)
    scene_simple = renderer_simple.scene
    scene_simple.add_geometry("mesh", mesh, material)
    
    scene_simple.set_background(np.array([0.4, 0.4, 0.4, 1.0], dtype=np.float32)) 
    scene_simple.set_lighting(scene_simple.LightingProfile.SOFT_SHADOWS, (0.5, 0.5, 0.5)) 
    scene_simple.scene.enable_sun_light(True)
    scene_simple.scene.set_sun_light(
        direction=[-1, -1, 1],
        color=[1, 1, 1],
        intensity=110000
    )
    
    # --- Normal Renderer ---
    normal_renderer = o3d.visualization.rendering.OffscreenRenderer(w, h)
    mesh.compute_vertex_normals()

    mat_normal = o3d.visualization.rendering.MaterialRecord()
    mat_normal.shader = "normals"
    mat_normal.base_roughness = 1
    mat_normal.base_metallic = 0.0
    mat_normal.base_reflectance = 0
    
    normal_renderer.scene.add_geometry("mesh", mesh, mat_normal)
    normal_renderer.scene.set_background(np.array([1, 1, 1, 1.0], dtype=np.float32))
    normal_renderer.scene.scene.enable_sun_light(False)
    normal_renderer.scene.view.set_post_processing(False)

    # 6. Output Path Configuration
    loaded_iter = scene.loaded_iter
    if args.output_dir:
        render_mesh_path = args.output_dir
    else:
        # Match structure: model_path/train/ours_X/mesh
        # Script 2 usually puts things in "renders", Script 1 in "mesh". We stick to Script 1's logic here.
        render_mesh_path = os.path.join(args.model_path, "train", f"ours_{loaded_iter}{args.suffix}", "mesh")
    
    os.makedirs(render_mesh_path, exist_ok=True)
    
    # Coordinate conversion matrix (COLMAP -> OpenGL logic from Script 1)
    colmap_to_opengl = np.eye(4)
    colmap_to_opengl[1, 1] = -1
    colmap_to_opengl[2, 2] = -1

    print(f"Starting rendering for {len(views)} views...")

    # 7. Rendering Loop
    for idx, view in enumerate(tqdm(views)):
        # Handle resolution changes if dataset has varying image sizes
        if view.image_width != w or view.image_height != h:
            # Note: For strict correctness with varying sizes, renderers should be re-initialized 
            # or use a max size. Here we assume constant size or skip resize to keep it simple/fast.
            pass 

        # Build Camera Pose (C2W)
        # Script 2 (GS) Camera has:
        # view.R: World-to-Camera Rotation Matrix
        # view.camera_center: Camera Position (C2W translation)
        pose = np.eye(4)
        pose[:3, :3] = view.R.transpose() # Transpose W2C rotation to get C2W rotation
        pose[:3, 3] = view.T

        # Apply the conversion to match the flipped mesh coordinates
        pose = pose @ colmap_to_opengl

        # Intrinsic Matrix
        # Script 2 Camera objects have Fx, Fy, Cx, Cy attributes
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=view.image_width,
            height=view.image_height,
            fx=view.Fx,
            fy=view.Fy,
            cx=view.Cx,
            cy=view.Cy
        )
    
        # Render Simple Pass
        renderer_simple.setup_camera(intrinsic, pose)
        image = renderer_simple.render_to_image()
        o3d.io.write_image(os.path.join(render_mesh_path, view.image_name + "_simple.png"), image)
        
        # Render Normal Pass
        normal_renderer.setup_camera(intrinsic, pose)
        world_normal_image = normal_renderer.render_to_image()
        o3d.io.write_image(os.path.join(render_mesh_path, view.image_name + "_normal.png"), world_normal_image)

    del renderer_simple
    del normal_renderer

    print("Rendering complete.")