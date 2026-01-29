# adapted from https://github.com/zju3dv/PGSR

import torch
from planargs.scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from planargs.gaussian_renderer import render
import torchvision
from planargs.common_utils.general_utils import safe_state
from argparse import ArgumentParser
from planargs.arguments import ModelParams, PipelineParams, PriorParams, get_combined_args
from planargs.gaussian_renderer import GaussianModel
import numpy as np
import cv2
import open3d as o3d
import copy
from planargs.planar.visualize import visualDepth, visualNorm


# Clean up isolated small pieces in the mesh.
def clean_mesh(mesh, min_len=1000):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_len
    mesh_0 = copy.deepcopy(mesh)
    mesh_0.remove_triangles_by_mask(triangles_to_remove)

    return mesh_0


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, max_depth=5.0, volume=None):
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    render_depthcolor_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_depthcolor")
    render_normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_normal")
    save_depthpath = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_depth")
    
    makedirs(gts_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(render_depthcolor_path, exist_ok=True)
    makedirs(render_normal_path, exist_ok=True)
    makedirs(save_depthpath, exist_ok=True)

    depths_tsdf_fusion = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt = view.gt_image
        out = render(view, gaussians, pipeline, background)
        rendering = out["render"].clamp(0.0, 1.0)
        _, H, W = rendering.shape

        depth = out["plane_depth"]
        depth_tsdf = depth.clone()
        visualDepth(depth, render_depthcolor_path, view.image_name[0])
        visualNorm(out["rendered_normal"], render_normal_path, view.image_name[0])

        if name == 'test':
            torchvision.utils.save_image(gt.clamp(0.0, 1.0), os.path.join(gts_path, view.image_name[0] + ".png"))
            torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name[0] + ".png"))
        else:
            rendering_np = (rendering.permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
            cv2.imwrite(os.path.join(render_path, view.image_name[0] + ".jpg"), rendering_np)
            np.save(os.path.join(save_depthpath, view.image_name[0] + ".npy"), depth_tsdf.cpu().numpy())

        depths_tsdf_fusion.append(depth_tsdf.squeeze().cpu())
        
    torch.cuda.empty_cache()
    if volume is not None:
        depths_tsdf_fusion = torch.stack(depths_tsdf_fusion, dim=0)
        for idx, view in enumerate(tqdm(views, desc="TSDF Fusion progress")):
            ref_depth = depths_tsdf_fusion[idx].cuda()
            ref_depth[ref_depth > max_depth] = 0
            ref_depth = ref_depth.detach().cpu().numpy()
            
            pose = np.identity(4)
            pose[:3,:3] = view.R.transpose(-1,-2)
            pose[:3, 3] = view.T
            color = o3d.io.read_image(os.path.join(render_path, view.image_name[0] + ".jpg"))
            depth = o3d.geometry.Image((ref_depth * 1000).astype(np.uint16))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_scale = 1000.0, depth_trunc = max_depth, convert_rgb_to_intensity=False)
            volume.integrate(
                rgbd,
                o3d.camera.PinholeCameraIntrinsic(W, H, view.K[0, 0], view.K[1, 1], view.K[0, 2], view.K[1, 2]),
                pose)
            


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, prior : PriorParams, 
                skip_train : bool, skip_test : bool, max_depth : float, voxel_size : float):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, prior, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        print(f"TSDF voxel_size {voxel_size}")
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length = voxel_size,
        sdf_trunc = 4.0 * voxel_size,
        color_type = o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), 
                       gaussians, pipeline, background, max_depth=max_depth, volume=volume)
            print("extract_triangle_mesh")
            mesh = volume.extract_triangle_mesh()

            path = os.path.join(dataset.model_path, "mesh")
            os.makedirs(path, exist_ok=True)
            
            o3d.io.write_triangle_mesh(os.path.join(path, "tsdf_fusion.ply"), mesh, 
                                       write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
            mesh = clean_mesh(mesh)
            mesh.remove_unreferenced_vertices()
            mesh.remove_degenerate_triangles()
            o3d.io.write_triangle_mesh(os.path.join(path, "tsdf_fusion_post.ply"), mesh, 
                                       write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)


if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    prior = PriorParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--voxel_size", default=0.02, type=float)
    parser.add_argument("--max_depth", default=100.0, type=float)

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), prior.extract(args), args.skip_train, args.skip_test, args.max_depth, args.voxel_size)
