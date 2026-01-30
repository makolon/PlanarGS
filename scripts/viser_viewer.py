"""
Interactive viewer for PlanarGS using Viser
Web-based real-time 3D visualization of trained Gaussian models
Based on gsplat's viewer implementation
"""

import time
from typing import Tuple
from argparse import ArgumentParser

import torch
import numpy as np
import viser
from nerfview import CameraState, Viewer

from planargs.arguments import ModelParams, PipelineParams, PriorParams, get_combined_args
from planargs.common_utils.general_utils import safe_state
from planargs.gaussian_renderer import render
from planargs.scene import Scene, GaussianModel
from planargs.common_utils.graphics_utils import getWorld2View2, getProjectionMatrix


def main():
    parser = ArgumentParser(description="Viser-based interactive viewer for PlanarGS")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    prior = PriorParams(parser)

    parser.add_argument("--iteration", default=-1, type=int,
                       help="Checkpoint iteration to load (-1 for latest)")
    parser.add_argument("--port", type=int, default=8080,
                       help="Port for web viewer")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host address to bind to")
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

    print("\n" + "=" * 60)
    print("PlanarGS Viser Viewer")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Source path: {args.source_path}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    gaussians = GaussianModel(args.sh_degree if hasattr(args, 'sh_degree') else 3)
    scene = Scene(model.extract(args), gaussians, prior.extract(args),
                 load_iteration=args.iteration, shuffle=False)

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pipe = pipeline.extract(args)

    print(f"Loaded model from iteration: {scene.loaded_iter}")
    print(f"Number of Gaussians: {len(gaussians.get_xyz)}")

    device = torch.device("cuda")

    # Define render function compatible with nerfview
    @torch.no_grad()
    def viewer_render_fn(camera_state: CameraState, img_wh: Tuple[int, int]):
        """Render function called by nerfview when camera moves"""
        width, height = img_wh
        c2w = camera_state.c2w  # camera to world matrix [4, 4]
        K = camera_state.get_K(img_wh)  # intrinsic matrix [3, 3]

        # Convert to torch
        c2w_t = torch.from_numpy(c2w).float().to(device)
        K_t = torch.from_numpy(K).float().to(device)

        # Get world to camera (view matrix)
        w2c = torch.inverse(c2w_t)  # [4, 4]

        # Compute FoV from intrinsics
        focal_x = float(K_t[0, 0].item())
        focal_y = float(K_t[1, 1].item())
        FoVx = 2 * np.arctan(width / (2 * focal_x))
        FoVy = 2 * np.arctan(height / (2 * focal_y))

        # Create simple camera object
        class SimpleCamera:
            pass

        custom_camera = SimpleCamera()
        custom_camera.image_width = width
        custom_camera.image_height = height
        custom_camera.FoVx = FoVx
        custom_camera.FoVy = FoVy
        # Build view/projection using PlanarGS conventions
        w2c_np = w2c.detach().cpu().numpy()
        R = w2c_np[:3, :3]
        T = w2c_np[:3, 3]

        world_view = getWorld2View2(R, T)
        custom_camera.world_view_transform = (
            torch.tensor(world_view).transpose(0, 1).to(device)
        )

        znear = 0.01
        zfar = 100.0
        proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy)
        custom_camera.full_proj_transform = (
            custom_camera.world_view_transform.unsqueeze(0)
            .bmm(proj.transpose(0, 1).to(device).unsqueeze(0))
            .squeeze(0)
        )
        custom_camera.camera_center = custom_camera.world_view_transform.inverse()[3, :3]

        # Render
        render_pkg = render(
            custom_camera,
            gaussians,
            pipe,
            background,
            return_plane=False,
            return_depth_normal=False
        )

        # Get RGB output
        output = render_pkg["render"]  # [3, H, W]
        output = torch.clamp(output, 0.0, 1.0)
        image = (output.permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8)  # [H, W, 3]

        return image

    # Start viser server and viewer
    print(f"\nStarting Viser server on {args.host}:{args.port}...")
    server = viser.ViserServer(host=args.host, port=args.port, verbose=False)

    Viewer(
        server=server,
        render_fn=viewer_render_fn,
        mode="rendering",
    )
    print("\n" + "=" * 60)
    print(f"Viewer is running at http://localhost:{args.port}")
    print("Open this URL in your web browser")
    print("Use mouse to navigate the 3D view")
    print("Press Ctrl+C to exit")
    print("=" * 60 + "\n")

    # Keep server running
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down viewer...")


if __name__ == "__main__":
    main()
