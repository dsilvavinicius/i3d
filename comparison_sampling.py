#!/usr/bin/env python
# coding: utf-8

"""
Comparison of different sampling approaches' impact on the performance of
machine learning models.
"""

import math
import os
import os.path as osp
import time
import pandas as pd
from plyfile import PlyData
import numpy as np
import open3d as o3d
import open3d.core as o3c
import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector
import diff_operators
from loss_functions import true_sdf, sdf_sitzmann
from meshing import (convert_sdf_samples_to_ply, gen_mc_coordinate_grid,
                     save_ply)
from model import SIREN


EPOCHS = 500
N_TEST_POINTS = 5000
BATCH_SIZE = 20000
SEED = 271668
N_RUNS = 10
CURVATURE_FRACS = (0.2, 0.6, 0.2)
LOW_MED_PERCENTILES = (70, 95)
METHODS = ["i3d"]

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def lowMedHighCurvSegmentation(
        mesh: o3d.t.geometry.TriangleMesh,
        n_samples: int,
        bin_edges: np.array,
        proportions: np.array,
        exceptions: list = []
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

    if exceptions:
        index = torch.Tensor(
            list(set(range(on_surface_pts.shape[0])) - set(exceptions)),
        ).int()
        on_surface_pts = torch.index_select(
            on_surface_pts, dim=0, index=index
        )

    curvatures = on_surface_pts[..., -1]

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


def sample_on_surface(mesh: o3d.t.geometry.TriangleMesh,
                      n_points: int,
                      exceptions: list = []) -> (torch.Tensor, np.ndarray):
    """Samples points from a mesh surface.

    Slightly modified from `i3d.dataset._sample_on_surface`. Returns the
    indices of points on surface as well and excludes points with indices in
    `exceptions`.

    Parameters
    ----------
    mesh: o3d.t.geometry.TriangleMesh
        The mesh to sample vertices from.

    n_points: int
        The number of vertices to sample.

    exceptions: list, optional
        The list of vertices to exclude from the selection. The default value
        is an empty list, meaning that any vertex might be selected. This works
        by setting the probabilities of any vertices with indices in
        `exceptions` to 0 and adjusting the probabilities of the remaining
        points.

    Returns
    -------
    samples: torch.Tensor
        The samples drawn from `mesh`

    idx: list
        The index of `samples` in `mesh`. Might be fed as input to further
        calls of `sample_on_surface`

    See Also
    --------
    numpy.random.choice
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
            curvature_fracs,
            on_surf_exceptions
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

    Note that we expect the input ply to contain x,y,z vertex data, as well
    as nx,ny,nz normal data and the curvature stored in the `quality` vertex
    property.

    Parameters
    ----------
    path: str, PathLike
        Path to the ply file. We except the file to be in binary format.

    Returns
    -------
    mesh: o3d.t.geometry.TriangleMesh
        The fully constructed Open3D Triangle Mesh. By default, the mesh is
        allocated on the CPU:0 device.

    vertices: numpy.array
        The same vertex information as stored in `mesh` returned for
        convenience only.

    See Also
    --------
    PlyData.read, o3d.t.geometry.TriangleMesh
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
        Total time to run the model inference. Does not include time to run
        marching cubes itself.

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


def grad_sdf(p: torch.Tensor, model: torch.nn.Module,
             no_curv: bool = False) -> (torch.Tensor, torch.Tensor):
    """Evaluates the gradient of F (`model`) at points `p`.

    Parameters
    ----------
    p: torch.Tensor
        The input points to feed to `model`.

    model: torch.nn.Module
        Neural network to calculate the gradient. Note that *the network must
        be differentiable*.

    no_curv: boolean, optional
        If set to True, we will not return the curvatures of `model` at `p`.
        The default value is False, meaning that curvatures will be calculated.
        If not needed, set it to True, since this calculation might take some
        time.

    Returns
    -------
    gradient: torch.Tensor
        A tensor with the gradient of `model` at points `p`.

    curvatures: torch.Tensor, None
        The curvatures of `model` at points `p`. Set to `None` if `no_curv`
        is True.
    """
    sdf = model(p)
    coords = sdf['model_in']
    values = sdf['model_out'].unsqueeze(0)
    gradient = diff_operators.gradient(values, coords)
    return gradient, diff_operators.divergence(gradient, coords) if not no_curv else None


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
    "happy_buddha": {
        "hidden_layer_config": [256, 256, 256, 256],
        "w0": 60,
        "ww": None,
    },
    "lucy": {
        "hidden_layer_config": [256, 256, 256, 256],
        "w0": 60,
        "ww": None,
    },
    "bunny": {
        "hidden_layer_config": [256, 256, 256],
        "w0": 30,
        "ww": None,
    },
    "dragon": {
        "hidden_layer_config": [256, 256, 256],
        "w0": 30,
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
    "bunny": osp.join("data", "bunny_curvs.ply"),
    "happy_buddha": osp.join("data", "happy_buddha_curvs.ply"),
    "dragon": osp.join("data", "dragon_curvs.ply"),
    "lucy": osp.join("data", "lucy_simple_curvs.ply"),
}

MESH_ORDER = ["armadillo", "bunny", "happy_buddha", "dragon", "lucy"]

for MESH_TYPE in MESH_ORDER:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    results_path = osp.join("comparison_results_sampling", MESH_TYPE)
    if not osp.exists(results_path):
        os.makedirs(results_path)

    mesh, vertices = read_ply(mesh_map[MESH_TYPE])

    # Create a raycasting scene to perform the SDF querying
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)

    test_surf_pts, test_surf_idx = sample_on_surface(
        mesh,
        N_TEST_POINTS // 2,
    )
    test_surf_pts = torch.from_numpy(test_surf_pts.numpy())

    test_domain_pts = np.random.uniform(
        [-1, -1, -1], [1, 1, 1],
        (N_TEST_POINTS // 2, 3)
    )
    test_domain_pts = o3c.Tensor(test_domain_pts, dtype=o3c.Dtype.Float32)
    test_domain_sdf = scene.compute_signed_distance(test_domain_pts)
    test_domain_sdf = torch.from_numpy(test_domain_sdf.numpy())
    test_domain_pts = torch.from_numpy(test_domain_pts.numpy())

    test_pts = torch.row_stack((
        test_surf_pts[..., :3],
        test_domain_pts
    ))
    test_normals = torch.row_stack((
        test_surf_pts[..., 3:6],
        torch.zeros_like(test_domain_pts)
    ))
    test_sdf = torch.cat((
        torch.zeros(test_surf_pts.shape[0]),
        torch.from_numpy(test_domain_sdf.numpy())
    ))

    l1, l2 = np.percentile(vertices[:, -1], [p for p in LOW_MED_PERCENTILES])
    CURVATURE_THRESHS = [
        np.min(vertices[:, -1]),
        l1,
        l2,
        np.max(vertices[:, -1])
    ]

    N = vertices.shape[0]

    if "siren" in METHODS:
        netconfig = netconfig_map.get(MESH_TYPE, netconfig_map["default"])
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)

        N_TRAINING_POINTS = N - N_TEST_POINTS
        N_STEPS = round(EPOCHS * (2 * N_TRAINING_POINTS / BATCH_SIZE))
        print(f"# of steps: {N_STEPS}")

        training_stats = {
            "mean_abs_error": [-1] * N_RUNS,
            "max_abs_error": [-1] * N_RUNS,
            "mean_abs_error_on_surface": [-1] * N_RUNS,
            "max_abs_error_on_surface": [-1] * N_RUNS,
            "mean_abs_error_off_surface": [-1] * N_RUNS,
            "max_abs_error_off_surface": [-1] * N_RUNS,
            "mean_normal_alignment": [-1] * N_RUNS,
            "max_normal_alignment": [-1] * N_RUNS,
            "execution_times": [-1] * N_RUNS
        }
        i = 0
        while i < N_RUNS:
            training_loss = {}
            model = SIREN(3, 1, **netconfig).cuda()
            print(model)
            print("# parameters =", parameters_to_vector(model.parameters()).numel())
            optim = torch.optim.Adam(lr=1e-4, params=model.parameters())

            # Start of the training loop
            for s in range(N_STEPS):
                pts, normals, sdf = create_training_data(
                    mesh=mesh,
                    n_on_surf=BATCH_SIZE // 2,
                    on_surf_exceptions=test_surf_idx,
                    n_off_surf=BATCH_SIZE // 2,
                    domain_bounds=[[-1, -1, -1], [1, 1, 1]],
                    scene=scene,
                    use_curvature=False,
                    no_sdf=True
                )

                pts.cuda()
                normals.cuda()
                sdf.cuda()

                gt = {
                    "sdf": sdf.float().unsqueeze(1),
                    "normals": normals.float(),
                }

                optim.zero_grad()

                y = model(pts[:, :3])
                loss = sdf_sitzmann(y, gt)

                running_loss = torch.zeros((1, 1)).cuda()
                for k, v in loss.items():
                    running_loss += v
                    if k not in training_loss:
                        training_loss[k] = [v.item()]
                    else:
                        training_loss[k].append(v.item())

                running_loss.backward()
                optim.step()

                if not s % 100 and s > 0:
                    print(f"Step {s} --- Loss {running_loss.detach().item()}")

            model.eval().cpu()
            n_siren, curv_siren = grad_sdf(test_surf_pts[..., :3], model)
            with torch.no_grad():
                y_siren = model(test_pts)["model_out"].squeeze()
                errs = torch.abs(test_sdf - y_siren)
                errs_on_surf = errs[:test_surf_pts.shape[0]]
                errs_off_surf = errs[test_surf_pts.shape[0]:]

                errs_normals = 1 - F.cosine_similarity(
                    test_normals[:test_surf_pts.shape[0], ...],
                    n_siren,
                    dim=-1
                )

            print(f"SIREN Results:"
                  f" MABSE {errs.mean():.3} -- MAXERR {errs.max().item():.3}")
            print(f"Errors on surface --- "
                  f"MABSE {errs_on_surf.mean():.3} -- MAXERR {errs_on_surf.max().item():.3}")
            print(f"Errors off surface --- "
                  f"MABSE {errs_off_surf.mean():.3} -- MAXERR {errs_off_surf.max().item():.3}")
            print(f"Normal alignment errors --- MEAN {errs_normals.mean():.3}"
                  f" --- MAX {errs_normals.max().item():.3}")

            training_stats["max_abs_error"][i] = errs.max().item()
            training_stats["mean_abs_error"][i] = errs.mean().item()
            training_stats["max_abs_error_on_surface"][i] = errs_on_surf.max().item()
            training_stats["mean_abs_error_on_surface"][i] = errs_on_surf.mean().item()
            training_stats["max_abs_error_off_surface"][i] = errs_off_surf.max().item()
            training_stats["mean_abs_error_off_surface"][i] = errs_off_surf.mean().item()
            training_stats["max_normal_alignment"][i] = errs_normals.max().item()
            training_stats["mean_normal_alignment"][i] = errs_normals.mean().item()

            i += 1

        verts, faces, total_time = run_marching_cubes(samples, model, MC_RESOLUTION)
        print(f"Marching cubes inference time {total_time:.3} s")
        save_ply(verts, faces, osp.join(results_path, "mc_siren.ply"))
        torch.save(model.state_dict(), osp.join(results_path, "weights_siren.pth"))
        stats_df = pd.DataFrame.from_dict(training_stats)
        stats_df.to_csv(
            osp.join(results_path, "stats_siren.csv"), sep=";", index=False
        )
        stats_df.mean().to_csv(
            osp.join(results_path, "stats_siren_mean.csv"), sep=";"
        )
        stats_df.std().to_csv(
            osp.join(results_path, "stats_siren_stddev.csv"), sep=";"
        )

    if "i3d" in METHODS:
        netconfig = netconfig_map.get(MESH_TYPE, netconfig_map["default"])
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)

        N_TRAINING_POINTS = N - N_TEST_POINTS
        N_STEPS = round(EPOCHS * (2 * N_TRAINING_POINTS / BATCH_SIZE))
        print(f"# of steps: {N_STEPS}")

        training_stats = {
            "mean_abs_error": [-1] * N_RUNS,
            "max_abs_error": [-1] * N_RUNS,
            "mean_abs_error_on_surface": [-1] * N_RUNS,
            "max_abs_error_on_surface": [-1] * N_RUNS,
            "mean_abs_error_off_surface": [-1] * N_RUNS,
            "max_abs_error_off_surface": [-1] * N_RUNS,
            "mean_normal_alignment": [-1] * N_RUNS,
            "max_normal_alignment": [-1] * N_RUNS,
            "execution_times": [-1] * N_RUNS
        }
        i = 0
        while i < N_RUNS:
            training_loss = {}
            model = SIREN(3, 1, **netconfig).cuda()
            print(model)
            print("# parameters =", parameters_to_vector(model.parameters()).numel())
            optim = torch.optim.Adam(lr=1e-4, params=model.parameters())

            # Start of the training loop
            for s in range(N_STEPS):
                pts, normals, sdf = create_training_data(
                    mesh=mesh,
                    n_on_surf=BATCH_SIZE // 2,
                    on_surf_exceptions=test_surf_idx,
                    n_off_surf=BATCH_SIZE // 2,
                    domain_bounds=[[-1, -1, -1], [1, 1, 1]],
                    scene=scene,
                    use_curvature=True,
                    curvature_fracs=CURVATURE_FRACS,
                    curvature_threshs=CURVATURE_THRESHS
                )

                pts.cuda()
                normals.cuda()
                sdf.cuda()

                gt = {
                    "sdf": sdf.float().unsqueeze(1),
                    "normals": normals.float(),
                }

                optim.zero_grad()

                y = model(pts[:, :3])
                loss = true_sdf(y, gt)

                running_loss = torch.zeros((1, 1)).cuda()
                for k, v in loss.items():
                    running_loss += v
                    if k not in training_loss:
                        training_loss[k] = [v.item()]
                    else:
                        training_loss[k].append(v.item())

                running_loss.backward()
                optim.step()

                if not s % 100 and s > 0:
                    print(f"Step {s} --- Loss {running_loss.detach().item()}")

            model.eval().cpu()
            n_i3d, curv_i3d = grad_sdf(test_surf_pts[..., :3], model)
            with torch.no_grad():
                y_i3d = model(test_pts)["model_out"].squeeze()
                errs = torch.abs(test_sdf - y_i3d)
                errs_on_surf = errs[:test_surf_pts.shape[0]]
                errs_off_surf = errs[test_surf_pts.shape[0]:]

                errs_normals = 1 - F.cosine_similarity(
                    test_normals[:test_surf_pts.shape[0], ...],
                    n_i3d,
                    dim=-1
                )

            print(f"i3d Results:"
                  f" MABSE {errs.mean():.3} -- MAXERR {errs.max().item():.3}")
            print(f"Errors on surface --- "
                  f"MABSE {errs_on_surf.mean():.3} -- MAXERR {errs_on_surf.max().item():.3}")
            print(f"Errors off surface --- "
                  f"MABSE {errs_off_surf.mean():.3} -- MAXERR {errs_off_surf.max().item():.3}")
            print(f"Normal alignment errors --- MEAN {errs_normals.mean():.3}"
                  f" --- MAX {errs_normals.max().item():.3}")

            training_stats["max_abs_error"][i] = errs.max().item()
            training_stats["mean_abs_error"][i] = errs.mean().item()
            training_stats["max_abs_error_on_surface"][i] = errs_on_surf.max().item()
            training_stats["mean_abs_error_on_surface"][i] = errs_on_surf.mean().item()
            training_stats["max_abs_error_off_surface"][i] = errs_off_surf.max().item()
            training_stats["mean_abs_error_off_surface"][i] = errs_off_surf.mean().item()
            training_stats["max_normal_alignment"][i] = errs_normals.max().item()
            training_stats["mean_normal_alignment"][i] = errs_normals.mean().item()

            i += 1

        verts, faces, total_time = run_marching_cubes(samples, model, MC_RESOLUTION)
        print(f"Marching cubes inference time {total_time:.3} s")
        save_ply(verts, faces, osp.join(results_path, f"mc_i3d_biasedcurvs.ply"))
        torch.save(model.state_dict(), osp.join(results_path, f"weights_i3d_biasedcurvs.pth"))
        stats_df = pd.DataFrame.from_dict(training_stats)
        stats_df.to_csv(
            osp.join(results_path, "stats_i3d.csv"), sep=";", index=False
        )
        stats_df.mean().to_csv(
            osp.join(results_path, "stats_i3d_mean.csv"), sep=";"
        )
        stats_df.std().to_csv(
            osp.join(results_path, "stats_i3d_stddev.csv"), sep=";"
        )
