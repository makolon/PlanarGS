"""
Orchestrator script that chains the PlanarGS pipeline:
1) DUSt3R geometric priors (run_geomprior.py)
2) LP3 planar prompts & masks (run_lp3.py)
3) PlanarGS training (train.py)
4) Optional rendering / TSDF fusion (render.py)

This keeps the per-stage CLIs intact while letting you run the full flow
with a single command.
"""
import os
import torch
from argparse import ArgumentParser

from arguments import (
    ModelParams,
    PipelineParams,
    OptimizationParams,
    PriorParams,
    get_combined_args,
)
from common_utils.general_utils import safe_state
from run_geomprior import GeomPrior
from run_lp3 import LP3
from train import training
from render import render_sets


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
                        default=[6999, 13999, 19999])
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
