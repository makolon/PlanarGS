"""
Interactive viewer for PlanarGS using Rerun
Web-based real-time 3D visualization of trained Gaussian models
"""
import torch
import numpy as np
from argparse import ArgumentParser

import rerun as rr
from planargs.arguments import ModelParams, PipelineParams, PriorParams, get_combined_args
from planargs.common_utils.general_utils import safe_state
from planargs.gaussian_renderer import render
from planargs.scene import Scene, GaussianModel


def visualize_gaussians(gaussians: GaussianModel, iteration: int):
    """Visualize Gaussian points in Rerun"""
    
    # Get Gaussian parameters
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    
    # Get colors (from spherical harmonics DC component)
    features_dc = gaussians._features_dc.detach()
    colors_sh = features_dc[:, 0, :]  # Take DC component (0th order)
    # Convert from SH to RGB (simplified)
    C0 = 0.28209479177387814
    colors = (colors_sh * C0 + 0.5).clamp(0, 1).cpu().numpy()
    
    # Get sizes (scaling)
    scales = gaussians.get_scaling.detach().cpu().numpy()
    # Use mean scale as point size
    point_sizes = scales.mean(axis=1)
    
    # Log points to Rerun
    rr.log(
        f"gaussians/iteration_{iteration}",
        rr.Points3D(
            positions=xyz,
            colors=colors,
            radii=point_sizes,
        )
    )
    
    print(f"Visualized {len(xyz)} Gaussians")


def render_and_log_camera(camera, gaussians, pipeline, background, camera_idx: int):
    """Render from a camera and log to Rerun"""
    
    with torch.no_grad():
        # Render
        render_pkg = render(
            camera,
            gaussians,
            pipeline,
            background,
            return_plane=True,
            return_depth_normal=True
        )
        
        # Get rendered image
        image = render_pkg["render"]
        image = torch.clamp(image, 0.0, 1.0)
        image_np = (image.permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8)
        
        # Get depth
        depth = render_pkg["plane_depth"]
        depth_np = depth.cpu().numpy()
        
        # Get normal
        normal = render_pkg["rendered_normal"]
        normal_np = ((normal.permute(1, 2, 0) + 1) / 2 * 255).cpu().numpy().astype(np.uint8)
        
        # Log camera pose
        R = camera.R
        T = camera.T
        
        # Convert to camera-to-world transform
        R_c2w = R.T
        t_c2w = -R_c2w @ T
        
        # Create 4x4 transform matrix
        transform = np.eye(4)
        transform[:3, :3] = R_c2w
        transform[:3, 3] = t_c2w
        
        # Log camera
        rr.log(
            f"cameras/camera_{camera_idx}",
            rr.Transform3D(
                transform=transform,
            )
        )
        
        # Log images
        rr.log(
            f"cameras/camera_{camera_idx}/image",
            rr.Image(image_np)
        )
        
        rr.log(
            f"cameras/camera_{camera_idx}/depth",
            rr.DepthImage(depth_np)
        )
        
        rr.log(
            f"cameras/camera_{camera_idx}/normal",
            rr.Image(normal_np)
        )


def main():
    parser = ArgumentParser(description="Rerun-based interactive viewer for PlanarGS")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    prior = PriorParams(parser)
    
    parser.add_argument("--iteration", default=-1, type=int,
                       help="Checkpoint iteration to load (-1 for latest)")
    parser.add_argument("--port", type=int, default=9876,
                       help="Port for Rerun web viewer")
    parser.add_argument("--render_cameras", action="store_true",
                       help="Render views from training cameras")
    parser.add_argument("--quiet", action="store_true")
    
    args = get_combined_args(parser)
    
    # Validate required arguments
    if not args.model_path:
        parser.error("Please provide --model_path/-m pointing to the trained model folder.")
    if not args.source_path:
        parser.error("Please provide --source_path/-s pointing to the COLMAP scene folder.")
    
    # Initialize system
    safe_state(args.quiet)
    torch.set_num_threads(8)
    
    # Setup output file
    output_file = f"{args.model_path}/visualization.rrd"
    
    # Initialize Rerun and save to file
    rr.init("PlanarGS Viewer", spawn=False)
    rr.save(output_file)
    
    print("\n" + "=" * 60)
    print("PlanarGS Rerun Viewer")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Source path: {args.source_path}")
    print(f"Output file: {output_file}")
    print("=" * 60)
    
    # Load model
    with torch.no_grad():
        gaussians = GaussianModel(args.sh_degree if hasattr(args, 'sh_degree') else 3)
        scene = Scene(model.extract(args), gaussians, prior.extract(args), 
                     load_iteration=args.iteration, shuffle=False)
        
        print(f"\nLoaded model from iteration: {scene.loaded_iter}")
        print(f"Number of Gaussians: {len(gaussians.get_xyz)}")
        
        # Visualize Gaussians
        print("\nVisualizing Gaussians...")
        visualize_gaussians(gaussians, scene.loaded_iter)
        
        # Optionally render from training cameras
        if args.render_cameras:
            print("\nRendering from training cameras...")
            bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            
            train_cameras = scene.getTrainCameras()
            for idx, camera in enumerate(train_cameras[:10]):  # Render first 10 cameras
                print(f"  Rendering camera {idx + 1}/10...", end='\r')
                render_and_log_camera(camera, gaussians, pipeline.extract(args), 
                                    background, idx)
            print("\nDone rendering cameras!")
        
        print("\n" + "=" * 60)
        print(f"Visualization saved to: {output_file}")
        print("\nTo view the recording:")
        print(f"  rerun {output_file}")
        print("\nOr upload it to https://rerun.io/viewer")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
