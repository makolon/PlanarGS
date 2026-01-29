"""
Orchestrator script that chains the PlanarGS pipeline:
1) DUSt3R geometric priors
2) LP3 planar prompts & masks
3) PlanarGS training
4) Optional rendering / TSDF fusion (render.py)

Stages 1-3 are consolidated here; rendering remains in render.py.
"""
import os
import sys
import random
from random import randint
from argparse import ArgumentParser

# Add submodules to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
dust3r_path = os.path.join(project_root, "submodules", "dust3r")
if dust3r_path not in sys.path:
    sys.path.insert(0, dust3r_path)

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from planargs.arguments import (
    ModelParams,
    PipelineParams,
    OptimizationParams,
    PriorParams,
    get_combined_args,
)
from planargs.common_utils.general_utils import safe_state
from planargs.common_utils.graphics_utils import get_k
from planargs.common_utils.loss_utils import l1_loss, l2_loss, ssim
from planargs.gaussian_renderer import render
from planargs.geomprior.dataloader import GroupAlign
from planargs.geomprior.run_dust3r import DUSt3R
from planargs.lp3.box_migrate import AddPreviosBox, FilterMask, LP3Cam
from planargs.lp3.color_cluster import MaskDistance
from planargs.lp3.mask_refine import BoxSmaller, NormalSplit
from planargs.lp3.run_groundedsam import GroundingDINO, SAM
from planargs.planar.co_planar import co_planar
from planargs.planar.training_report import prepare_output_and_logger, training_report
from planargs.planar.visualize import visualSegmask, visualMask
from planargs.scene import Scene, GaussianModel
from planargs.scene.cameras import LoadGeomprior
from planargs.scene.dataset_readers import readColmapSceneInfo
from render import render_sets

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'


def get_numeric_part(filename):
    return int(''.join(filter(str.isdigit, filename)))


# Sampling-based grouping is used to ensure that each image group covers the entire scene.
def GroupFiles(data_dir, output_dir, ckpt, group_size, vis):
    files = sorted([f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))],
                   key=get_numeric_part)

    total_files = len(files)
    num_groups = (total_files + group_size - 1) // group_size
    groups = [[] for _ in range(num_groups)]

    for i, file in enumerate(files):
        group_idx = i % num_groups
        groups[group_idx].append(file)

    for idx, group in enumerate(groups, 1):
        group_folder = os.path.join(output_dir, f"_group{idx}")
        os.makedirs(group_folder, exist_ok=True)
        DUSt3R(data_dir, group_folder, ckpt, group, vis)  # Set vis=True to save the reconstructed dense point cloud from DUSt3R.

    print("Finish depth predicting.")


def GeomPrior(model, prep, group_size, vis, skip_model, skip_align):
    datapath = model.source_path
    ckpt = prep.ckpt_mv
    gp_data_path = os.path.join(datapath, "geomprior")
    os.makedirs(gp_data_path, exist_ok=True)
    image_path = os.path.join(datapath, "images")
    # depth generation
    if not skip_model:
        GroupFiles(image_path, gp_data_path, ckpt, group_size, vis)
    # depth align and resize
    if not skip_align:
        if os.path.exists(os.path.join(datapath, "sparse")):
            scene_info = readColmapSceneInfo(datapath, eval=False)
            cam_infos = scene_info.train_cameras
            GroupAlign(prep, cam_infos, scene_info.points3d, gp_data_path, vis)  # Set vis=True to visualize prior depth & normal
        else:
            assert False, "Could not recognize scene type!"


def draw_boxes(image, boxes, labels):
    """
    Args:
        image: np.ndarray, HWC, BGR
        boxes: list of [x0, y0, x1, y1]
        labels: list of str
    Returns:
        image_with_boxes: np.ndarray
    """
    img = image.copy()

    for box, label in zip(boxes, labels):
        x0, y0 = int(box[0]), int(box[1])
        x1, y1 = int(box[2]), int(box[3])
        color = (0, 255, 0) if not label.startswith("[proj]") else (0, 0, 255)  # BGR
        thickness = 2
        cv2.rectangle(img, (x0, y0), (x1, y1), color=color, thickness=thickness)

        # draw texts
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        text_thickness = 2
        _, baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
        cv2.putText(img, label, (x0, y0 + 2 * baseline), font, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)

    return img


def LP3(model, prp, text_prompts, vis, device="cuda"):   # Set vis=True to visualize boxes & masks.
    input_folder = model.source_path
    output_folder = os.path.join(input_folder, "planarprior")
    os.makedirs(output_folder, exist_ok=True)
    if prp.visdebug:
        debug_folder = os.path.join(output_folder, "debug")
        os.makedirs(debug_folder, exist_ok=True)

    detection = GroundingDINO(device)
    detection.load_model(prp.ckpt_det)

    segmentation = SAM(device)
    segmentation.load_model(prp.ckpt_seg)

    if os.path.exists(os.path.join(input_folder, "sparse")):
        scene_info = readColmapSceneInfo(input_folder, eval=False)
    else:
        assert False, "Could not recognize scene type!"

    cam_infos = scene_info.train_cameras
    add_previous = False
    previous_cam = None

    for cam in tqdm(cam_infos, desc="Processing cameras"):
        file_path = os.path.join(cam.path, "images", cam.image_name[0] + cam.image_name[1])
        image_name = cam.image_name[0]
        geom_folder = os.path.join(cam.path, "geomprior")
        depth, normal = LoadGeomprior(geom_folder, image_name, cam.size)
        W, H = cam.size
        K, inv_K = get_k(cam.FovX, cam.FovY, H, W)
        if prp.visdebug:
            camdebug_folder = os.path.join(debug_folder, image_name)
            os.makedirs(camdebug_folder, exist_ok=True)
        else:
            camdebug_folder = None

        detection.load_image(file_path)
        boxes_filt, pred_phrases = detection.get_detection_output(text_prompts, with_logits=False)

        # The box is represented by its top-left and bottom-right vertices.
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        if add_previous:
            boxes_filt, pred_phrases = AddPreviosBox(pred_phrases, boxes_filt, cam, previous_cam, camdebug_folder)
        boxes_filt, pred_phrases = BoxSmaller(boxes_filt, pred_phrases)

        # Segmentation from boxes
        segmentation.load_image(file_path)
        masks = segmentation.get_segmentation_mask(boxes_filt)

        distance_mask = MaskDistance(depth, normal, inv_K, prp.dis_thresh)
        if prp.visdebug:
            visualMask(distance_mask, path=camdebug_folder, filename="distance")

        masks = masks * distance_mask
        masks, pred_phrases, boxes_filt = NormalSplit(masks, pred_phrases, boxes_filt, normal, prp.normal_split, camdebug_folder)

        masks_previous, pred_previous = FilterMask(masks, pred_phrases)
        add_previous = True
        previous_cam = LP3Cam(cam_info=cam, depth=depth, preds=pred_previous, masks=masks_previous)

        mask_folder = os.path.join(output_folder, "mask")
        os.makedirs(mask_folder, exist_ok=True)
        num = 1
        mask_use = torch.zeros(masks[0].shape).to(device)
        for mask in masks:
            mask_use = mask_use + mask * num
            num += 1
        np.save(os.path.join(mask_folder, image_name + ".npy"), mask_use.cpu().numpy())

        if vis:
            image = cv2.cvtColor(np.array(Image.open(file_path)), cv2.COLOR_RGB2BGR)
            vis_boxes_path = os.path.join(output_folder, "boxes_vis")
            vis_mask_path = os.path.join(output_folder, "mask_vis")
            os.makedirs(vis_boxes_path, exist_ok=True)
            os.makedirs(vis_mask_path, exist_ok=True)
            image_with_boxes = draw_boxes(image, boxes_filt, pred_phrases)
            cv2.imwrite(os.path.join(vis_boxes_path, image_name + ".png"), image_with_boxes)
            visualSegmask(mask_use.squeeze(0), path=vis_mask_path, filename=image_name)   # image can be added as background


def training(dataset, opt, pipe, prp, test_iters, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, prp)
    gaussians.training_setup(opt)
    viewpoint_stack = scene.getTrainCameras().copy()

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    # Planar-guided initialization
    elif prp.use_initdensify:
        densify_num = prp.initdensify_num
        initdensify_idx = 0
        cam_num = len(viewpoint_stack)
        densify_image_list = []
        pbar = tqdm(total=densify_num, desc="Planar-guided initialization")
        # Densify only 50 perspectives to avoid out-of-memory issues
        if cam_num > densify_num:
            densify_num_list = random.sample(range(0, cam_num), densify_num)
            num = 0
            for cam in viewpoint_stack:
                if num in densify_num_list:
                    densify_image_list.append(cam.image_name)
                num += 1
        else:
            for cam in viewpoint_stack:
                densify_image_list.append(cam.image_name)
            densify_num = cam_num

        for idx in range(cam_num):
            viewpoint_cam = viewpoint_stack[idx]
            if viewpoint_cam.image_name in densify_image_list:
                initdensify_idx += 1
                render_pkg = render(viewpoint_cam, gaussians, pipe, background, return_plane=False, return_depth_normal=False)
                vis_mask = (render_pkg["out_observe"] > 0) & render_pkg["visibility_filter"]
                gaussians.plane_initdensify(viewpoint_cam, vis_mask, prp)
                pbar.update(1)
                if (initdensify_idx == densify_num):
                    pbar.close()
                    print("\nFinish densifying gaussian in plane area.")
                    point_cloud_path = os.path.join(dataset.model_path, "point_cloud/planar_initdensify")
                    print(f"Saving Gaussians to {point_cloud_path}")
                    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
                    break

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    ema_imageloss_for_log = 0.0
    ema_dnloss_for_log = 0.0
    ema_planar_for_log = 0.0
    ema_priordep_for_log = 0.0
    ema_priornorm_for_log = 0.0

    plane_loss, prior_deploss, prior_normloss, dn_loss = None, None, None, None

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        gt_image, prior_depth, prior_normal = viewpoint_cam.gt_image, viewpoint_cam.priordepth, viewpoint_cam.priornormal
        canny_mask, planarmasks, conf_mask = viewpoint_cam.canny_mask, viewpoint_cam.planarmask, viewpoint_cam.depth_conf
        # Default: no planar region; gets overwritten when a valid planar mask exists
        planar_mask = torch.zeros_like(canny_mask)

        # Iterations for depth/normal rendering
        iter_argu = min(opt.dnloss_iteration, opt.planar_iteration, opt.priordepth_iteration)
        iter_dn = min(opt.dnloss_iteration, opt.priornormal_iteration)
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg,
                            return_plane=(iteration >= iter_argu), return_depth_normal=(iteration >= iter_dn))
        image, visibility_filter, radii = render_pkg["render"], render_pkg["visibility_filter"], render_pkg["radii"]
        out_observe, viewspace_point_tensor_abs, viewspace_point_tensor = render_pkg["out_observe"], render_pkg["viewspace_points_abs"], render_pkg["viewspace_points"]
        if iteration >= iter_argu:
            render_depth, render_normal = render_pkg["plane_depth"], render_pkg["rendered_normal"]
        if iteration >= iter_dn:
            depth_normal = render_pkg["depth_normal"]

        ### Loss
        ssim_loss = (1.0 - ssim(image, gt_image))
        Ll1 = l1_loss(image, gt_image)
        image_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
        loss = image_loss.clone()

        # scale loss
        if visibility_filter.sum() > 0:
            scale = gaussians.get_scaling[visibility_filter]
            sorted_scale, _ = torch.sort(scale, dim=-1)
            min_scale_loss = sorted_scale[..., 0]
            loss += opt.lambda_scale * min_scale_loss.mean()

        # depth prior loss
        if iteration >= opt.priordepth_iteration and prior_depth is not None:
            depth_mask = (prior_depth > 0)
            depth_prior_mask = canny_mask * conf_mask
            prior_deploss = l2_loss(prior_depth, render_depth * depth_mask, depth_prior_mask) * viewpoint_cam.depth_weight * opt.lambda_priordepth
            loss = loss + prior_deploss

        # planar loss
        if iteration >= opt.planar_iteration and planarmasks is not None:
            if planarmasks.max() != 0:
                planar_depth = co_planar(render_depth, planarmasks, viewpoint_cam.inv_K)
                planar_mask = (planarmasks > 0).int()
                plane_loss = opt.lambda_planar * l1_loss(render_depth, planar_depth, planar_mask)
                loss += plane_loss

            # prior normal loss
            if iteration >= opt.priornormal_iteration and prior_normal is not None:
                ones = torch.ones((prior_normal.shape[1], prior_normal.shape[2])).cuda()
                prior_normloss_1 = l1_loss(prior_normal, depth_normal, planar_mask * depth_prior_mask)
                prior_normloss_2 = torch.mean(((ones - torch.sum(prior_normal * depth_normal, dim=0)) * planar_mask * depth_prior_mask))
                prior_normloss = opt.lambda_priornormal * (prior_normloss_1 + prior_normloss_2)
                loss += prior_normloss

        # normal-consistency loss
        if iteration > opt.dnloss_iteration:
            if iteration >= opt.planar_iteration:
                dn_loss = opt.lambda_dnloss * l1_loss(render_normal, depth_normal, (1 - planar_mask) * canny_mask)
            else:
                dn_loss = opt.lambda_dnloss * l1_loss(render_normal, depth_normal, canny_mask)
            loss += dn_loss

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_imageloss_for_log = 0.4 * image_loss.item() + 0.6 * ema_imageloss_for_log
            ema_dnloss_for_log = 0.4 * dn_loss.item() + 0.6 * ema_dnloss_for_log if dn_loss is not None else 0.0
            ema_planar_for_log = 0.4 * plane_loss.item() + 0.6 * ema_planar_for_log if plane_loss is not None else 0.0
            ema_priordep_for_log = 0.4 * prior_deploss.item() + 0.6 * ema_priordep_for_log if prior_deploss is not None else 0.0
            ema_priornorm_for_log = 0.4 * prior_normloss.item() + 0.6 * ema_priornorm_for_log if prior_normloss is not None else 0.0

            if iteration % 10 == 0:
                loss_dict = {
                    "ImageLoss": f"{ema_imageloss_for_log:.{5}f}",
                    "DNloss": f"{ema_dnloss_for_log:.{5}f}",
                    "Planeloss": f"{ema_planar_for_log:.{5}f}",
                    "PriorDeploss": f"{ema_priordep_for_log:.{5}f}",
                    "PriorNormloss": f"{ema_priornorm_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, dn_loss, plane_loss, prior_deploss, prior_normloss, co_planar,
                            l1_loss, iter_start.elapsed_time(iter_end), test_iters, scene, render, (pipe, background), vis_planar=(iteration >= opt.planar_iteration))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                mask = (out_observe > 0) & visibility_filter
                gaussians.max_radii2D[mask] = torch.max(gaussians.max_radii2D[mask], radii[mask])
                gaussians.add_densification_stats(viewspace_point_tensor, viewspace_point_tensor_abs, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_abs_grad_threshold,
                                                opt.opacity_cull_threshold, scene.cameras_extent, size_threshold)

            # reset_opacity
            if iteration < opt.densify_until_iter:
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            if iteration % 500 == 0:
                torch.cuda.empty_cache()


def build_parser():
    parser = ArgumentParser(description="End-to-end runner for PlanarGS pipeline")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)
    prior = PriorParams(parser)

    # GeomPrior
    parser.add_argument("--group_size", type=int, default=40, help="images per DUSt3R group")
    parser.add_argument("--geomprior_vis", action="store_true", help="visualize DUSt3R outputs")
    parser.add_argument("--skip_geomprior", action="store_true", help="skip geometric prior stage")
    parser.add_argument("--skip_geom_model", action="store_true", help="reuse existing DUSt3R predictions")
    parser.add_argument("--skip_geom_align", action="store_true", help="skip depth align/resize")

    # LP3
    parser.add_argument("-t", "--text_prompts", type=str,
                        default="wall. floor. door. screen. window. ceiling. table",
                        help="period-separated prompt list")
    parser.add_argument("--lp3_vis", action="store_true", help="save LP3 boxes/masks visualizations")
    parser.add_argument("--skip_lp3", action="store_true", help="skip LP3 stage")
    parser.add_argument("--device", type=str, default="cuda", help="device for LP3 models")

    # Training (defaults mirror train.py)
    parser.add_argument("--skip_train", action="store_true", help="skip training stage")
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[1, 7000, 14000, 20000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int,
                        default=[7000, 14000, 20000, 30000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int,
                        default=[6999, 13999, 19999, 29999])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--detect_anomaly", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--threads", type=int, default=8, help="torch.set_num_threads value")

    # Rendering (optional)
    parser.add_argument("--run_render", action="store_true", help="run render.py after training")
    parser.add_argument("--render_iteration", type=int, default=-1,
                        help="checkpoint iteration to render (-1 = latest)")
    parser.add_argument("--skip_render_train", action="store_true",
                        help="render only test set")
    parser.add_argument("--skip_render_test", action="store_true",
                        help="render only train set / mesh")
    parser.add_argument("--voxel_size", default=0.02, type=float, help="TSDF voxel size (meters)")
    parser.add_argument("--max_depth", default=100.0, type=float, help="depth truncation for TSDF")

    return parser, model, pipeline, opt, prior


def validate_paths(args, parser):
    if not args.source_path:
        parser.error("Please provide --source_path/-s pointing to the COLMAP scene folder.")
    if not args.model_path:
        parser.error("Please provide --model_path/-m for outputs/checkpoints.")
    if not os.path.isdir(args.source_path):
        parser.error(f"source_path does not exist: {args.source_path}")
    os.makedirs(args.model_path, exist_ok=True)


def main():
    parser, model_group, pipeline_group, opt_group, prior_group = build_parser()
    args = get_combined_args(parser)
    validate_paths(args, parser)

    # Match original scripts’ runtime settings
    safe_state(args.quiet)
    torch.set_num_threads(args.threads)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    model = model_group.extract(args)
    pipeline = pipeline_group.extract(args)
    opt = opt_group.extract(args)
    prior = prior_group.extract(args)

    if not args.skip_geomprior:
        print("\n[Stage 1/4] Geometric priors (DUSt3R)")
        GeomPrior(model, prior, args.group_size, args.geomprior_vis,
                  args.skip_geom_model, args.skip_geom_align)
    else:
        print("\n[Stage 1/4] Skipped Geometric priors")

    if not args.skip_lp3:
        print("\n[Stage 2/4] LP3 planar prompts & masks")
        LP3(model, prior, args.text_prompts, args.lp3_vis, device=args.device)
    else:
        print("\n[Stage 2/4] Skipped LP3")

    if not args.skip_train:
        print("\n[Stage 3/4] PlanarGS training")
        training(model, opt, pipeline, prior,
                 args.test_iterations, args.save_iterations,
                 args.checkpoint_iterations, args.start_checkpoint)
    else:
        print("\n[Stage 3/4] Skipped training")

    if args.run_render:
        print("\n[Stage 4/4] Rendering / TSDF fusion")
        render_sets(model, args.render_iteration, pipeline, prior,
                    args.skip_render_train, args.skip_render_test,
                    args.max_depth, args.voxel_size)
    else:
        print("\n[Stage 4/4] Rendering skipped (use --run_render to enable)")


if __name__ == "__main__":
    main()
