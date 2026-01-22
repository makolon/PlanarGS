#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary
from common_utils.graphics_utils import focal2fov
from scene.ply_loader import SceneInfo, CameraInfo, getNerfppNorm, storePly, fetchPly
import numpy as np
import json
import torch
from typing import Optional


def load_poses(pose_path, num):
    poses = []
    with open(pose_path, "r") as f:
        lines = f.readlines()
    for i in range(num):
        line = lines[i]
        c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
        c2w[:3,3] = c2w[:3,3] * 10.0
        w2c = np.linalg.inv(c2w)
        w2c = w2c
        poses.append(w2c)
    poses = np.stack(poses, axis=0)
    return poses


def SingleReadColmap(idx, key, cam_extrinsics, cam_intrinsics, path):
    sys.stdout.write('\r')
     # the exact output you're looking for:
    sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
    sys.stdout.flush()

    extr = cam_extrinsics[key]
    intr = cam_intrinsics[extr.camera_id]
    height = intr.height 
    width = intr.width 

    uid = intr.id  
    R = np.transpose(qvec2rotmat(extr.qvec))
    T = np.array(extr.tvec)

    if intr.model=="SIMPLE_PINHOLE" or intr.model=="SIMPLE_RADIAL":
        focal_length_x = intr.params[0]
        focal_length_y = intr.params[0]
        FovY = focal2fov(focal_length_x, height)
        FovX = focal2fov(focal_length_x, width)
    elif intr.model=="PINHOLE":
        focal_length_x = intr.params[0]
        focal_length_y = intr.params[1]
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)
    else:
        assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

    image_name = os.path.splitext(extr.name)
    points3d_ids = extr.point3D_ids
        
    cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX,
                        path=path, image_name=image_name,
                        size=(width, height), points3d_ids=points3d_ids)
        
    return cam_info


def readColmapCameras(cam_extrinsics, cam_intrinsics, path):
    cam_infos = []

    for idx, key in enumerate(cam_extrinsics): 
        cam_info = SingleReadColmap(idx, key, cam_extrinsics, cam_intrinsics, path)
        cam_infos.append(cam_info)
        torch.cuda.empty_cache()
    sys.stdout.write('\n')
    return cam_infos


def readColmapSceneInfo(path, eval, llffhold=8):  
    def _find_model_dir(sparse_root: str) -> Optional[str]:
        """Return a colmap model dir that contains images.(bin|txt).
        Priority: subdirs with images.bin > subdirs with images.txt > root with bin/txt."""

        def has_bin(dirpath: str) -> bool:
            return os.path.isfile(os.path.join(dirpath, "images.bin"))

        def has_txt(dirpath: str) -> bool:
            return os.path.isfile(os.path.join(dirpath, "images.txt"))

        subdirs = [os.path.join(sparse_root, d) for d in os.listdir(sparse_root)
                   if os.path.isdir(os.path.join(sparse_root, d))]
        def sort_key(p):
            name = os.path.basename(p)
            return (0, int(name)) if name.isdigit() else (1, name)
        subdirs = sorted(subdirs, key=sort_key)

        for d in subdirs:
            if has_bin(d):
                return d
        for d in subdirs:
            if has_txt(d):
                return d
        if has_bin(sparse_root) or has_txt(sparse_root):
            return sparse_root
        return None

    sparse_root = os.path.join(path, "sparse")
    model_dir = _find_model_dir(sparse_root)
    assert model_dir is not None, f"Could not find COLMAP model under {sparse_root}"

    xyz = rgb = points3d = None
    if os.path.isfile(os.path.join(model_dir, "images.bin")):
        cameras_extrinsic_file = os.path.join(model_dir, "images.bin")
        cameras_intrinsic_file = os.path.join(model_dir, "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    else:
        cameras_extrinsic_file = os.path.join(model_dir, "images.txt")
        cameras_intrinsic_file = os.path.join(model_dir, "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
            
    N = len([f for f in os.listdir(os.path.join(path, "images"))])
    
    js_file = f"{path}/train_test_lists.json"

    if eval:
        if os.path.exists(js_file):
            train_list = []
            test_list = []
            with open(js_file) as file:
                meta = json.load(file)
                train_list = [os.path.splitext(file)[0] for file in meta["train"]]
                test_list = [os.path.splitext(file)[0] for file in meta["test"]]
                print(f"train_list {len(train_list)}, test_list {len(test_list)}")

        else:
            train_list = [idx for idx in range(N) if idx % llffhold != 0]
            test_list = [idx for idx in range(N) if idx % llffhold == 0]
    else:
        train_list = [idx for idx in range(N)]


    ply_path = os.path.join(model_dir, "points3D.ply")
    bin_path = os.path.join(model_dir, "points3D.bin")
    pcd = None
    if os.path.exists(bin_path):
        xyz, rgb, points3d = read_points3D_binary(bin_path)
        try:
            print("Converting point3d.bin to .ply.")
            storePly(ply_path, xyz, rgb)
        except Exception as e:
            print(f"Warning: failed to store ply at {ply_path}: {e}")
        # Build in-memory point cloud even if ply write fails
        from common_utils.graphics_utils import BasicPointCloud
        pcd = BasicPointCloud(points=xyz, colors=rgb / 255.0 if rgb is not None else None, normals=None, errors=None)
    else:
        points3d = None

    if pcd is None:
        try:
            pcd = fetchPly(ply_path)
        except Exception as e:
            print(f"Warning: failed to load ply at {ply_path}: {e}")
    
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, path=path)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)  
    if eval:
        if isinstance(train_list[0], str):
            print("read train/test image names from json file.\n")
            train_cam_infos = []
            test_cam_infos = []
            for cam in cam_infos:
                if cam.image_name in train_list:
                    train_cam_infos.append(cam)
                elif cam.image_name in test_list:
                    test_cam_infos.append(cam)
                else:
                    print(f"can not find image in the split file: {cam.image_name}\n")
        else:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in train_list] 
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in test_list]   
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_info = SceneInfo(point_cloud=pcd,
                           points3d = points3d,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info
