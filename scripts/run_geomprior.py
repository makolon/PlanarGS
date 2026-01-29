import os
from argparse import ArgumentParser

from planargs.geomprior.run_dust3r import DUSt3R
from planargs.arguments import ModelParams, PriorParams
from planargs.geomprior.dataloader import GroupAlign
from planargs.scene.dataset_readers import readColmapSceneInfo


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


if __name__ == '__main__':
    parser = ArgumentParser(description="Generate geometric priors script parameters")
    model = ModelParams(parser, sentinel=True)
    prp = PriorParams(parser)
    parser.add_argument('--group_size', type=str,help='number of images in each group', default=40)
    parser.add_argument("--vis", action="store_true") 
    parser.add_argument("--skip_model", action="store_true")
    parser.add_argument("--skip_align", action="store_true")
    args = parser.parse_args()

    GeomPrior(model.extract(args), prp.extract(args), int(args.group_size), args.vis, args.skip_model, args.skip_align)
