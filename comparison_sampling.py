#!/usr/bin/env python
# coding: utf-8

"""Comparison of different sampling approaches impact on the performance of machine learning models
"""

import math
import os
import os.path as osp
import time
from plyfile import PlyData
import numpy as np
import open3d as o3d
import open3d.core as o3c
import torch
from torch.nn.utils import parameters_to_vector
import diff_operators
from loss_functions import true_sdf, sdf_sitzmann
from meshing import (convert_sdf_samples_to_ply, gen_mc_coordinate_grid,
                     save_ply)
from model import SIREN


EPOCHS = 10
N_TEST_POINTS = 0
BATCH_SIZE = 20000
SEED = 271668

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def lowMedHighCurvSegmentation(
        mesh: o3d.t.geometry.TriangleMesh,
        n_samples: int,
        bin_edges: np.array,
        proportions: np.array
):
    """Samples `n_points` points from `mesh` based on their curvature.

    This function is based on `i3d.dataset.lowMedHighCurvSegmentation`.

    Parameters
    ----------
    mesh: o3d.t.geometry.TriangleMesh,
        The mesh to sample points from.

    n_samples: int
        Number of samples to fetch.

    bin_edges: np.array
        The [minimum, low-medium threshold, medium-high threshold, maximum]
        curvature values in `mesh`. These values define thresholds between low
        and medium curvature values, and medium to high curvatures.

    proportions: np.array
        The percentage of points to fetch for each curvature band per batch of
        `n_samples`.

    Returns
    -------
    samples: torch.Tensor
        The vertices sampled from `mesh`.
    """
    on_surface_sampled = 0
    on_surface_pts = torch.column_stack((
        torch.from_numpy(mesh.vertex["positions"].numpy()),
        torch.from_numpy(mesh.vertex["normals"].numpy()),
        torch.from_numpy(mesh.vertex["curvatures"].numpy())
    ))

    curvatures = torch.from_numpy(mesh.vertex["curvatures"].numpy())

    low_curvature_pts = on_surface_pts[(curvatures >= bin_edges[0]) & (curvatures < bin_edges[1]), ...]
    low_curvature_idx = np.random.choice(
        range(low_curvature_pts.shape[0]),
        size=int(math.floor(proportions[0] * n_samples)),
        replace=False
    )
    on_surface_sampled = len(low_curvature_idx)

    med_curvature_pts = on_surface_pts[(curvatures >= bin_edges[1]) & (curvatures < bin_edges[2]), ...]
    med_curvature_idx = np.random.choice(
        range(med_curvature_pts.shape[0]),
        size=int(math.ceil(proportions[1] * n_samples)),
        replace=False
    )
    on_surface_sampled += len(med_curvature_idx)

    high_curvature_pts = on_surface_pts[(curvatures >= bin_edges[2]) & (curvatures <= bin_edges[3]), ...]
    high_curvature_idx = np.random.choice(
        range(high_curvature_pts.shape[0]),
        size=n_samples - on_surface_sampled,
        replace=False
    )

    return torch.cat((
        low_curvature_pts[low_curvature_idx, ...],
        med_curvature_pts[med_curvature_idx, ...],
        high_curvature_pts[high_curvature_idx, ...]
    ), dim=0)


def sample_on_surface(mesh,
                      n_points: int,
                      exceptions: list = []) -> (torch.Tensor, np.ndarray):
    """Samples points from a mesh surface.

    Slightly modified from `i3d.dataset._sample_on_surface`. Returns the
    indices of points on surface as well.
    """
    if exceptions:
        p = np.array(
            [1. / (len(mesh.vertex["positions"]) - len(exceptions))] *
            len(mesh.vertex["positions"])
        )
        p[exceptions] = 0.0

    idx = np.random.choice(
        np.arange(start=0, stop=len(mesh.vertex["positions"])),
        size=n_points,
        replace=False,
        p=p if exceptions else None
    )
    on_points = mesh.vertex["positions"].numpy()[idx]
    if "normals" not in mesh.vertex:
        on_normals = np.zeros_like(on_points)
        print("No normals found. Marking all as zeroes.")
    else:
        on_normals = mesh.vertex["normals"].numpy()[idx]

    return torch.from_numpy(np.hstack((
        on_points,
        on_normals,
        np.zeros((n_points, 1))
    )).astype(np.float32)), idx.tolist()


def create_training_data(
        mesh: o3d.t.geometry.TriangleMesh,
        n_on_surf: int,
        on_surf_exceptions: list,
        n_off_surf: int,
        domain_bounds,
        scene: o3d.t.geometry.RaycastingScene,
        no_sdf: bool = False,
        use_curvature: bool = False,
        curvature_fracs: list = [],
        curvature_threshs: list = [],
):
    """Creates a set of training data with coordinates, normals and SDF
    values.

    Parameters
    ----------
    mesh: o3d.t.geometry.TriangleMesh
        A Tensor-backed Open3D mesh.

    n_on_surf: int
        # of points to sample from the mesh.

    on_surf_exceptions: list
        Points that cannot be used for training, i.e. test set of points.

    n_off_surf: int
        # of points to sample from the domain. Note that we sample points
        uniformely at random from the domain.

    domain_bounds: tuple[np.array, np.array]
        Bounds to use when sampling points from the domain.

    scene: o3d.t.geometry.RaycastingScene
        Open3D raycasting scene to use when querying SDF for domain points.

    no_sdf: boolean, optional
        If using SIREN's original loss, we do not query SDF for domain
        points, instead we mark them with SDF = -1.

    use_curvature: boolean, optional
        Indicates if we must use the curvature to perform sampling on surface
        points.

    curvature_fracs: list, optional
        The fractions of points to sample per curvature band. Only used when
        `use_curvature` is True.

    curvature_threshs: list
        The curvature values to use when defining low, medium and high
        curvatures. Only used when `use_curvature` is True.

    Returns
    -------
    training_pts: torch.Tensor
    training_normals: torch.Tensor
    training_sdf: torch.Tensor

    See Also
    --------
    sample_on_surface, lowMedHighCurvSegmentation
    """
    if use_curvature:
        surf_pts = lowMedHighCurvSegmentation(
            mesh,
            n_on_surf,
            curvature_threshs,
            curvature_fracs
        )
    else:
        surf_pts, _ = sample_on_surface(
            mesh,
            n_on_surf,
            on_surf_exceptions
        )
    surf_pts = torch.from_numpy(surf_pts.numpy())

    domain_pts = np.random.uniform(
        domain_bounds[0], domain_bounds[1],
        (n_off_surf, 3)
    )

    if not no_sdf:
        domain_pts = o3c.Tensor(domain_pts, dtype=o3c.Dtype.Float32)
        domain_sdf = scene.compute_signed_distance(domain_pts)
        domain_sdf = torch.from_numpy(domain_sdf.numpy())
        domain_pts = torch.from_numpy(domain_pts.numpy())
    else:
        domain_pts = torch.from_numpy(domain_pts)
        domain_sdf = -1 * torch.ones(domain_pts.shape[0])

    domain_normals = torch.zeros_like(domain_pts)

    training_pts = torch.row_stack((
        surf_pts[..., :3],
        domain_pts
    ))
    training_normals = torch.row_stack((
        surf_pts[..., 3:6],
        domain_normals
    ))
    training_sdf = torch.cat((
        torch.zeros(len(surf_pts)),
        domain_sdf
    ))

    return training_pts.float(), training_normals.float(), training_sdf.float()


def read_ply(path: str):
    """Reads a PLY file with position, normal and curvature info.

    Parameters
    ----------
    path: str, PathLike
        Path to the ply file

    Returns
    -------
    mesh: o3d.t.geometry.TriangleMesh
    vertices: numpy.array
    """
    # Reading the PLY file with curvature info
    with open(path, "rb") as f:
        plydata = PlyData.read(f)
        num_verts = plydata["vertex"].count
        vertices = np.zeros(shape=(num_verts, 7), dtype=np.float32)
        vertices[:, 0] = plydata["vertex"].data["x"]
        vertices[:, 1] = plydata["vertex"].data["y"]
        vertices[:, 2] = plydata["vertex"].data["z"]
        vertices[:, 3] = plydata["vertex"].data["nx"]
        vertices[:, 4] = plydata["vertex"].data["ny"]
        vertices[:, 5] = plydata["vertex"].data["nz"]
        vertices[:, 6] = plydata["vertex"].data["quality"]

        faces = np.stack(plydata["face"].data["vertex_indices"])

    # Converting the PLY data to open3d format
    device = o3c.Device("CPU:0")
    mesh = o3d.t.geometry.TriangleMesh(device)
    mesh.vertex["positions"] = o3c.Tensor(vertices[:, :3], dtype=o3c.float32)
    mesh.vertex["normals"] = o3c.Tensor(vertices[:, 3:6], dtype=o3c.float32)
    mesh.vertex["curvatures"] = o3c.Tensor(vertices[:, -1], dtype=o3c.float32)
    mesh.triangle["indices"] = o3c.Tensor(faces, dtype=o3c.int32)

    return mesh, vertices


def run_marching_cubes(samples: torch.Tensor, model, resolution: int):
    """Runs marching cubes on a set of samples using the SDF estimated by
    `model`.

    Parameters
    ----------
    samples: torch.Tensor
        The samples to evaluate `model` and run marching cubes.

    model: torch.nn.Module or scipy.interpolate.RBFInterpolator
        Any model with a __call__ or __forward__ method implemented.
        If `model` is derived of torch.nn.Module, you must call `model.eval()`
        first.

    resolution: int
        The marching cubes resolution

    Returns
    -------
    verts:
    faces:
    tot_time: float

    See Also
    --------
    gen_mc_coordinate_grid, convert_sdf_samples_to_ply
    """
    start = time.time()
    if isinstance(model, torch.nn.Module):
        samples_sdf = model(samples[:, :3])["model_out"].reshape([resolution] * 3).detach()
    else:
        samples_sdf = model(samples[:, :3]).reshape([resolution] * 3)
    tot_time = time.time() - start
    verts, faces, _, _ = convert_sdf_samples_to_ply(
        samples_sdf, [-1, -1, -1], voxel_size, None, None
    )
    return verts, faces, tot_time


# Marching cubes inputs
MC_RESOLUTION = 64
voxel_size = 2.0 / (MC_RESOLUTION - 1)
samples = gen_mc_coordinate_grid(MC_RESOLUTION, voxel_size)


netconfig_map = {
    "armadillo": {
        "hidden_layer_config": [256, 256, 256, 256],
        "w0": 60,
        "ww": None,
    },
    "bunny": {
        "hidden_layer_config": [256, 256, 256],
        "w0": 60,
        "ww": None,
    },
    "default": {
        "hidden_layer_config": [256, 256, 256],
        "w0": 30,
        "ww": None,
    }
}

mesh_map = {
    "armadillo": osp.join("data", "armadillo_curvs.ply"),
    "bunny": osp.join("data", "bunny_curvs.ply")
}

MESH_TYPE = "bunny"


results_path = osp.join("comparison_results_sampling", MESH_TYPE)
if not osp.exists(results_path):
    os.makedirs(results_path)

mesh, vertices = read_ply(mesh_map[MESH_TYPE])

# Creating SDF querying structures
scene = o3d.t.geometry.RaycastingScene()
_ = scene.add_triangles(mesh)

# Testing the querying
# N_TEST_POINTS = 5000
# test_domain_pts = np.random.uniform(
#     -1, 1,
#     (N_TEST_POINTS // 2, 3)
# )

# test_domain_pts = o3c.Tensor(test_domain_pts, dtype=o3c.Dtype.Float32)
# test_domain_sdf = scene.compute_signed_distance(test_domain_pts)
# test_domain_sdf = torch.from_numpy(test_domain_sdf.numpy())
# test_domain_pts = torch.from_numpy(test_domain_pts.numpy())

# From dataset.PointCloudCachedCurvature
CURVATURE_FRACS = (0.2, 0.6, 0.2)
LOW_MED_PERCENTILES = (70, 95)

l1, l2 = np.percentile(vertices[:, -1], [p for p in LOW_MED_PERCENTILES])
CURVATURE_THRESHS = [
    np.min(vertices[:, -1]),
    l1,
    l2,
    np.max(vertices[:, -1])
]

# on_surface_samples = lowMedHighCurvSegmentation(
#     torch.from_numpy(mesh.vertex["positions"].numpy()),
#     BATCH_SIZE // 2,
#     torch.from_numpy(mesh.vertex["curvatures"].numpy()),
#     CURVATURE_THRESHS, CURVATURE_FRACS
# )

N = vertices.shape[0]

## I3D
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

training_loss = {}
model = SIREN(3, 1, **netconfig_map.get(MESH_TYPE, netconfig_map["default"]))
print(model)
print("# parameters =", parameters_to_vector(model.parameters()).numel())
optim = torch.optim.Adam(lr=1e-4, params=model.parameters())

N_TRAINING_POINTS = N - N_TEST_POINTS
N_STEPS = round(EPOCHS * (2 * N_TRAINING_POINTS / BATCH_SIZE))
print(f"# of steps: {N_STEPS}")

for s in range(N_STEPS):
    pts, normals, sdf = create_training_data(
        mesh=mesh,
        n_on_surf=BATCH_SIZE // 2,
        on_surf_exceptions=[],
        n_off_surf=BATCH_SIZE // 2,
        domain_bounds=[[-1, -1, -1], [1, 1, 1]],
        scene=scene,
        use_curvature=True,
        curvature_fracs=CURVATURE_FRACS,
        curvature_threshs=CURVATURE_THRESHS
    )

    gt = {
        "sdf": sdf.float().unsqueeze(1),
        "normals": normals.float(),
    }

    optim.zero_grad()

    y = model(pts[:, :3])
    loss = true_sdf(y, gt)

    running_loss = torch.zeros((1, 1))
    for k, v in loss.items():
        running_loss += v
        if k not in training_loss:
            training_loss[k] = [v.item()]
        else:
            training_loss[k].append(v.item())

    running_loss.backward()
    optim.step()

    print(f"Step {s} --- Loss {running_loss.detach().item()}")

model.eval()
verts, faces, total_time = run_marching_cubes(samples, model, MC_RESOLUTION)
print(f"Marching cubes inference time {total_time:.3} s")
save_ply(verts, faces, osp.join(results_path, f"mc_i3d_{MESH_TYPE}_biasedcurvs.ply"))
torch.save(model.state_dict(), osp.join(results_path, f"weights_i3d_{MESH_TYPE}_biasedcurvs.pth"))

## SIREN
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

training_loss = {}
model = SIREN(3, 1, **netconfig_map.get(MESH_TYPE, netconfig_map["default"]))
print(model)
print("# parameters =", parameters_to_vector(model.parameters()).numel())
optim = torch.optim.Adam(lr=1e-4, params=model.parameters())

N_TRAINING_POINTS = N - N_TEST_POINTS
N_STEPS = round(EPOCHS * (2 * N_TRAINING_POINTS / BATCH_SIZE))
print(f"# of steps: {N_STEPS}")

for s in range(N_STEPS):
    pts, normals, sdf = create_training_data(
        mesh=mesh,
        n_on_surf=BATCH_SIZE // 2,
        on_surf_exceptions=[],
        n_off_surf=BATCH_SIZE // 2,
        domain_bounds=[[-1, -1, -1], [1, 1, 1]],
        scene=scene,
        use_curvature=False,
    )

    gt = {
        "sdf": sdf.float().unsqueeze(1),
        "normals": normals.float(),
    }

    optim.zero_grad()

    y = model(pts[:, :3])
    loss = sdf_sitzmann(y, gt)

    running_loss = torch.zeros((1, 1))
    for k, v in loss.items():
        running_loss += v
        if k not in training_loss:
            training_loss[k] = [v.item()]
        else:
            training_loss[k].append(v.item())

    running_loss.backward()
    optim.step()

    print(f"Step {s} --- Loss {running_loss.detach().item()}")

model.eval()
verts, faces, total_time = run_marching_cubes(samples, model, MC_RESOLUTION)
print(f"Marching cubes inference time {total_time:.3} s")
save_ply(verts, faces, osp.join(results_path, "mc_siren.ply"))
torch.save(model.state_dict(), osp.join(results_path, "weights_siren.pth"))
