#!/usr/bin/env python
# coding: utf-8


import argparse
import os
import numpy as np
import pyrender
import sys


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

colors = np.zeros((samples.shape[0], 3))
colors[samples[:, -1] < 0, 0] = 1
colors[samples[:, -1] > 0, 1] = 1

cloud = pyrender.Mesh.from_points(samples[:, :3], colors=colors)
scene = pyrender.Scene()
scene.add(cloud)
viewer = pyrender.Viewer(scene, use_raymond_lighting=True,
                         point_size=3)
