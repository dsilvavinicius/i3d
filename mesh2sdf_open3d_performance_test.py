#!/usr/bin/env python
# coding: utf-8

"""Script to benchmark the SDF calculation for Open3D and mesh-to-sdf."""

from mesh_to_sdf import get_surface_point_cloud
import numpy as np
import open3d as o3d
import trimesh
import timeit


nsamples = 64
nruns = 100
path = o3d.data.ArmadilloMesh().path

omesh = o3d.io.read_triangle_mesh(path)
omesh = o3d.t.geometry.TriangleMesh.from_legacy(omesh)

# Create a raycasting scene to perform the SDF querying
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(omesh)

# Creating a distribution of points in the mesh domain.
minbound = omesh.vertex["positions"].min(0).numpy()
maxbound = omesh.vertex["positions"].max(0).numpy()
coordrange = np.linspace(minbound, maxbound, num=nsamples)
querypts = np.stack(np.meshgrid(*coordrange.T), axis=-1).astype(np.float32)

tmesh = trimesh.Trimesh(vertices=omesh.vertex["positions"].numpy().astype(float),
                        faces=omesh.triangle["indices"].numpy().astype(int))
point_cloud = get_surface_point_cloud(tmesh, surface_point_method="scan",
                                      calculate_normals=True)

# query_points has shape [N, N, N, 3], we must reshape it to [N*N*N, 3]
# before passing to mesh-to-sdf
querypts_r = querypts.reshape(-1, 3)


def o3dsdf_calc():
    sdf = scene.compute_signed_distance(querypts)
    return sdf


def mesh2sdf_calc():
    sdf = point_cloud.get_sdf(querypts_r, use_depth_buffer=False,
                              return_gradients=False)
    return sdf


total_seconds_o3d = timeit.Timer(o3dsdf_calc).timeit(nruns)
avg_o3d = total_seconds_o3d / nruns
print("------Open3D------")
print(f"number of runs: {nruns}")
print(f"total time: {total_seconds_o3d} s")
print(f"average time {avg_o3d} s per run")

total_seconds_m2sdf = timeit.Timer(mesh2sdf_calc).timeit(nruns)
avg_m2sdf = total_seconds_m2sdf / nruns
print("------Mesh-to-SDF------")
print(f"number of runs: {nruns}")
print(f"total time: {total_seconds_m2sdf} s")
print(f"average time {avg_m2sdf} s per run")
