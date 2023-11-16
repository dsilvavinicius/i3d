#!/usr/bin/env python
# coding: utf-8

"""
Simple script to preprocess a mesh, normalizing it to the interval [-1, 1]^3, using Open3D.
"""

import argparse
import open3d as o3d
import os
import os.path as osp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", type=str, required=True, help="Path to the input mesh file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output mesh file")
    args = parser.parse_args()

    mesh_path = args.mesh_path
    output_path = args.output_path
    
    out_dir = mesh_path
    out_dir = osp.split(out_dir)[0]
    if out_dir and not osp.exists(out_dir):
        os.makedirs(out_dir)

    print("Normalizing the mesh")

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    print("Mesh: ", mesh)
    max_extent = mesh.get_axis_aligned_bounding_box().get_max_extent()
    mesh = mesh.translate(-mesh.get_axis_aligned_bounding_box().get_center())
    mesh = mesh.scale(2.0 / max_extent, (0.0, 0.0, 0.0))

    print('bbox: ', mesh.get_axis_aligned_bounding_box())

    o3d.io.write_triangle_mesh(output_path, mesh, write_vertex_normals=True, write_triangle_uvs=True, print_progress=True)

    print("Done")
