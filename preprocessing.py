#!/usr/bin/env python
# coding: utf-8


import argparse
import numpy as np
import os
from mesh_to_sdf import get_surface_point_cloud, scale_to_unit_sphere
from mesh_to_sdf.surface_point_cloud import SurfacePointCloud
from mesh_to_sdf.utils import sample_uniform_points_in_unit_sphere

import pyrender
import trimesh


parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "input_path",
    default="data/armadillo_mcurv.ply",
    help="Path to the input file/folder (if folder, then all files in it will be processed)."
)
parser.add_argument(
    "output_path",
    default=".",
    help="Output path."
)
parser.add_argument(
    "--samples_on_surface", "-s",
    type=int, default=1000,
    help="Number of samples to draw from the surface of the mesh (i.e. SDF = 0)."
)
parser.add_argument(
    "--samples_near_surface", "-n",
    type=int, default=1000,
    help="Number of samples to draw near the mesh surface."
)
parser.add_argument(
    "--samples_uniform", "-u",
    type=int, default=1000,
    help="Number of samples to draw uniformely at random from the mesh bounding box."
)
parser.add_argument(
    "--sample_vertices", "-S",
    action="store_true", help="Sample mesh vertices as surface points."
)
parser.add_argument(
    "--near_surface_vertices", "-N",
    action="store_true", help="Samples the near-surface points from mesh vertices."
)
parser.add_argument(
    "--sample_on_aabb", "-B",
    action="store_true", help="Samples points on the mesh AABB, instead of its bounding circle."
)
parser.add_argument(
    "--view_samples", "-V",
    action="store_true", help="Views the samples using pyrender."
)

args = parser.parse_args()

if os.path.isdir(args.input_path):
    for path in os.listdir(args.input_path):
        print(path)
else:
    full_samples = None
    mesh = trimesh.load(args.input_path)

    if args.samples_on_surface:
        # Surface sampling
        if args.sample_vertices:
            idx = np.random.choice(
                np.arange(start=0, stop=len(mesh.vertices)),
                size=args.samples_on_surface,
                replace=False
            )
            on_points = mesh.vertices[idx]
            on_normals = mesh.vertex_normals[idx]
        else:
            on_points, face_idx = mesh.sample(
                count=args.samples_on_surface,
                return_index=True
            )
            on_normals = mesh.face_normals[face_idx]

        full_samples = np.hstack((
            on_points,
            on_normals,
            np.zeros((args.samples_on_surface, 3))
        ))

    if args.samples_near_surface:
        # Near surface sampling
        point_cloud = get_surface_point_cloud(
            scale_to_unit_sphere(mesh),
            surface_point_method="scan",
            bounding_radius=1,
            calculate_normals=True
        )
        if args.near_surface_vertices:
            dist_samples = args.samples_near_surface  # int(args.samples_near_surface // 2)
            idx = np.random.choice(
                np.arange(start=0, stop=len(mesh.vertices)),
                size=dist_samples,
                replace=False
            )
            verts = mesh.vertices[idx]
            normals = mesh.vertex_normals[idx]
            scale = np.random.normal(scale=0.0025, size=dist_samples)
            displacement = np.multiply(normals, scale[:, np.newaxis])
            print(sum(scale < 0))
            print(sum(scale > 0))
            print(displacement)

            near_points = verts + displacement
        else:
            # From `SurfacePointCloud.sample_sdf_near_surface`. That method samples 6%
            # of points uniformely at random from the domain with no control of how
            # many points we want. Thus, we sample them manually later and constrain
            # the next lines to sample only points near the mesh surface.
            dist_samples = int(args.samples_near_surface // 2)
            rand_samples = point_cloud.get_random_surface_points(dist_samples)
            near_points = []
            near_points.append(
                rand_samples + np.random.normal(scale=0.0025, size=(dist_samples, 3))
            )
            near_points.append(
                rand_samples + np.random.normal(scale=0.00025, size=(dist_samples, 3))
            )
            near_points = np.concatenate(near_points).astype(float)

        near_sdf, near_normals = point_cloud.get_sdf(
            near_points,
            use_depth_buffer=False,
            return_gradients=True
        )

        print("# of samples inside mesh: ", sum(near_sdf < 0))
        print("# of samples outside mesh: ", sum(near_sdf > 0))

        near_samples = np.hstack((
            near_points,
            near_normals,
            np.zeros((args.samples_near_surface, 2)),
            near_sdf.reshape(args.samples_near_surface, 1)
        ))
        if full_samples is None:
            full_samples = near_samples
        else:
            full_samples = np.vstack((full_samples, near_samples))

    if args.samples_uniform:
        # Uniform sampling
        if args.sample_on_aabb:
            domain_points = np.random.uniform(-1, 1, size=(args.samples_uniform, 3))
        else:
            domain_points = sample_uniform_points_in_unit_sphere(args.samples_uniform)

        domain_sdf, domain_normals = point_cloud.get_sdf(
            domain_points,
            use_depth_buffer=False,
            return_gradients=True
        )

        domain_samples = np.hstack((
            domain_points,
            domain_normals,
            np.zeros((args.samples_uniform, 2)),
            domain_sdf.reshape((args.samples_uniform, 1))
        ))
        if full_samples is None:
            full_samples = domain_samples
        else:
            full_samples = np.vstack((full_samples, domain_samples))

    np.savetxt(args.output_path, full_samples)

    if args.view_samples:
        colors = None
        if args.samples_on_surface:
            colors = np.zeros((args.samples_on_surface, 3))

        if args.samples_near_surface:
            near_colors = np.zeros(near_points.shape)
            near_colors[near_sdf < 0, 0] = 1
            near_colors[near_sdf > 0, 1] = 1

            if colors is None:
                colors = near_colors
            else:
                colors = np.vstack((colors, near_colors))

        if args.samples_uniform:
            domain_colors = np.zeros(domain_points.shape)
            domain_colors[domain_sdf < 0, 0] = 1
            domain_colors[domain_sdf > 0, 1] = 1

            if colors is None:
                colors = domain_colors
            else:
                colors = np.vstack((colors, domain_colors))

        cloud = pyrender.Mesh.from_points(full_samples[:, :3], colors=colors)
        scene = pyrender.Scene()
        scene.add(cloud)
        viewer = pyrender.Viewer(scene, use_raymond_lighting=True,
                                 point_size=3)
