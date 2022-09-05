#!/usr/bin/env python
# coding: utf-8

"""
Script to calculate the mesh curvatures given an implicit representation
of it.
"""

import open3d as o3d
import numpy as np
import torch
import diff_operators
from util import from_pth
from meshing import save_ply


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
