#!/usr/bin/env python
# coding: utf-8


import argparse
import numpy as np
import os
from mesh_to_sdf import get_surface_point_cloud, scale_to_unit_sphere
from mesh_to_sdf.utils import sample_uniform_points_in_unit_sphere
import trimesh
import sys


parser = argparse.ArgumentParser(
    description="Program to sample points given a mesh. Outputs the results in XYZ format."
)
parser.add_argument(
    "input_path",
    help="Path to the input file. Note that the format must be supported by trimesh."
)
parser.add_argument(
    "output_path",
    help="Path to the output file"
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

args = parser.parse_args()

if not os.path.exists(args.input_path):
    print("[ERROR] Input path \"{args.input_path}\" does not exist.")
    sys.exit(1)

full_samples = None
mesh = trimesh.load(args.input_path)
mesh = scale_to_unit_sphere(mesh)

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

point_cloud = get_surface_point_cloud(
    mesh,
    surface_point_method="scan",
    bounding_radius=1,
    calculate_normals=True
)
if args.samples_near_surface:
    # Near surface sampling
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

    in_surf = domain_sdf < 0
    domain_sdf = np.concatenate((
        domain_sdf[in_surf],
        domain_sdf[~in_surf][:sum(in_surf)]), axis=0
    )
    domain_points = np.vstack((
        domain_points[in_surf, :],
        domain_points[~in_surf, :][:sum(in_surf)]
    ))
    domain_normals = np.vstack((
        domain_normals[in_surf, :],
        domain_normals[~in_surf, :][:sum(in_surf)]
    ))

    domain_samples = np.hstack((
        domain_points,
        domain_normals,
        np.zeros((domain_points.shape[0], 2)),
        domain_sdf[:, np.newaxis]
    ))

    # rm_idx = np.abs(domain_samples[:, -1]) > 5e-2
    # domain_samples = domain_samples[rm_idx, :]
    # domain_points = domain_points[rm_idx, :]
    # domain_sdf = domain_sdf[rm_idx]
    if full_samples is None:
        full_samples = domain_samples
    else:
        full_samples = np.vstack((full_samples, domain_samples))

np.savetxt(args.output_path, full_samples)
