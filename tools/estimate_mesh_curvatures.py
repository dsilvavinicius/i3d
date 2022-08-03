#!/usr/bin/env python
# coding: utf-8

"""
Script to calculate the mesh curvatures given an implicit representation
of it.
"""

import os.path as osp
import open3d as o3d
import numpy as np
import torch
import diff_operators
from model import SIREN
from util import siren_v1_to_v2
from meshing import save_ply


def from_pth(path, device="cpu", w0=None, ww=None):
    if not osp.exists(path):
        raise ValueError(f"Weights file not found at \"{path}\"")

    weights = torch.load(path, map_location=torch.device(device))
    # Each layer has two tensors, one for weights other for biases.
    n_layers = len(weights) // 2
    hidden_layer_config = [None] * (n_layers - 1)
    keys = list(weights.keys())

    bias_keys = [k for k in keys if "bias" in k]
    i = 0
    while i < (n_layers - 1):
        k = bias_keys[i]
        hidden_layer_config[i] = weights[k].shape[0]
        i += 1

    n_in_features = weights[keys[0]].shape[1]
    n_out_features = weights[keys[-1]].shape[0]
    model = SIREN(
        n_in_features=n_in_features,
        n_out_features=n_out_features,
        hidden_layer_config=hidden_layer_config,
        w0=w0, ww=ww
    )

    # Loads the weights. Converts to version 2 if they are from the old version
    # of SIREN.
    try:
        model.load_state_dict(weights)
    except RuntimeError:
        print("Found weights from old version of SIREN. Converting to v2.")
        new_weights, diff = siren_v1_to_v2(weights, True)
        new_weights_file = path.split(".")[0] + "_v2.pth"
        torch.save(new_weights, new_weights_file)
        model.load_state_dict(new_weights)

    return model


mesh_map = {
    "armadillo": ["./data/armadillo.ply", 60],
    "bunny": ["./data/bunny.ply", 30],
    "buddha": ["./data/happy_buddha.ply", 60],
    "dragon": ["./data/dragon.ply", 60],
    "lucy": ["./data/lucy_simple.ply", 60],
}

for MESH_TYPE in mesh_map.keys():
    mesh_path, w0 = mesh_map[MESH_TYPE]
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    print(mesh)

    coords = torch.from_numpy(mesh.vertex["positions"].numpy())

    model = from_pth(
        f"./results/{MESH_TYPE}_biased_curvature_sdf/models/model_best.pth",
        w0=w0
    ).eval()
    print(model)

    out = model(coords)
    X = out['model_in']
    y = out['model_out']

    curvatures = diff_operators.mean_curvature(y, X)
    verts = np.hstack((coords.detach().numpy(),
                       mesh.vertex["normals"].numpy(),
                       curvatures.detach().numpy()))
    faces = mesh.triangle["indices"].numpy()

    attrs = [("nx", "f4"), ("ny", "f4"), ("nz", "f4"), ("quality", "f4")]
    save_ply(verts, faces, f"./results/{MESH_TYPE}_calc_curvs.ply",
             vertex_attributes=attrs)
