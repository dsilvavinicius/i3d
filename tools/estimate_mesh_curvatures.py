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

    out = model(coords.unsqueeze(0))
    X = out['model_in']
    y = out['model_out']

    gradient = diff_operators.gradient(y,X)
    hessian  = diff_operators.hessian(y,X)

    min_curv,max_curv  = diff_operators.principal_curvature(y, X, gradient, hessian)
    min_dir ,max_dir   = diff_operators.principal_directions(gradient, hessian)

    # mean_curv = diff_operators.mean_curvature(y, X)
    mean_curv = (min_curv+max_curv)*0.5

    #vertices of the mesh with the directions of min curvatures and with the min curvatures  (x, v_min, k_min)
    verts_min = np.hstack((coords.squeeze(0).detach().numpy(),
                           min_dir.squeeze(0).detach().numpy(),
                           min_curv.squeeze(0).detach().numpy()))

    #vertices of the mesh with the directions of max curvatures and with the max curvatures  (x, v_max, k_max)
    verts_max = np.hstack((coords.squeeze(0).detach().numpy(),
                           max_dir.squeeze(0).detach().numpy(),
                           max_curv.squeeze(0).detach().numpy()))

    #vertices of the mesh with their normals and with the mean curvatures  (x, N, k_mean)
    verts_mean = np.hstack((coords.squeeze(0).detach().numpy(),
                            mesh.vertex["normals"].numpy(),
                            mean_curv.squeeze(0).detach().numpy()))

    faces = mesh.triangle["indices"].numpy()

    attrs = [("nx", "f4"), ("ny", "f4"), ("nz", "f4"), ("quality", "f4")]
    save_ply(verts_min, faces, f"./results/{MESH_TYPE}_min_curvs.ply",
             vertex_attributes=attrs)
    save_ply(verts_max, faces, f"./results/{MESH_TYPE}_max_curvs.ply",
            vertex_attributes=attrs)
    save_ply(verts_mean, faces, f"./results/{MESH_TYPE}_mean_curvs.ply",
            vertex_attributes=attrs)
