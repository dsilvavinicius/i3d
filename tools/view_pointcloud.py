#!/usr/bin/env python3
# coding: utf-8


import argparse
import os
import numpy as np
import pyrender
import sys


def toggle_mesh_vis(viewer, mesh):
    mesh.is_visible = not mesh.is_visible


parser = argparse.ArgumentParser(
    description="Simple renderer for the sampled point clouds."
)
parser.add_argument(
    "input_path",
    help="Path to the input point cloud (in XYZ format)."
)
args = parser.parse_args()

if not os.path.exists(args.input_path):
    print(f"[ERROR] Input path \"{args.input_path}\" does not exist.")
    sys.exit(1)

samples = np.loadtxt(args.input_path)

surf_samples = samples[samples[:, -1] == 0, :]
int_samples = samples[samples[:, -1] < 0, :]
ext_samples = samples[samples[:, -1] > 0, :]

print("Samples on surface: ", surf_samples.shape[0])
print("Samples with SDF < 0: ", int_samples.shape[0])
print("Samples with SDF > 0: ", ext_samples.shape[0])


scene = pyrender.Scene()
registered_keys = {}

if surf_samples.shape[0] > 0:
    surf_cloud = pyrender.Mesh.from_points(
        points=surf_samples[:, :3],
        normals=surf_samples[:, 3:6],
        colors=np.zeros((surf_samples.shape[0], 3))
    )
    scene.add(surf_cloud)
    registered_keys["u"] = (toggle_mesh_vis, [surf_cloud])

if int_samples.shape[0] > 0:
    int_cloud = pyrender.Mesh.from_points(
        points=int_samples[:, :3],
        normals=int_samples[:, 3:6],
        colors=np.zeros((int_samples.shape[0], 3)) + [1, 0, 0]
    )
    scene.add(int_cloud)
    registered_keys["i"] = (toggle_mesh_vis, [int_cloud])

if ext_samples.shape[0] > 0:
    ext_cloud = pyrender.Mesh.from_points(
        points=ext_samples[:, :3],
        normals=ext_samples[:, 3:6],
        colors=np.zeros((ext_samples.shape[0], 3)) + [0, 1, 0]
    )
    scene.add(ext_cloud)
    registered_keys["e"] = (toggle_mesh_vis, [ext_cloud])

light = pyrender.PointLight(intensity=500)

scene.add(light)
viewer = pyrender.Viewer(
    scene,
    point_size=2,
    use_raymond_lighting=True,
    registered_keys=registered_keys
)
