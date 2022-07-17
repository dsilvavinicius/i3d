#!/usr/bin/env python
# coding: utf-8


import time
import open3d as o3d
import open3d.core as o3c
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.interpolate import RBFInterpolator
from torch.nn.utils import parameters_to_vector
import diff_operators
from loss_functions import true_sdf, sdf_sitzmann
from meshing import (convert_sdf_samples_to_ply, gen_mc_coordinate_grid,
                     save_ply)
from model import SIREN


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
    on_normals = mesh.vertex["normals"].numpy()[idx]

    return torch.from_numpy(np.hstack((
        on_points,
        on_normals,
        np.zeros((n_points, 1))
    )).astype(np.float32)), idx.tolist()


def grad_sdf(p: torch.Tensor, model: torch.nn.Module,
             no_curv: bool = False) -> (torch.Tensor, torch.Tensor):
    """Evaluates the gradient of F (`model`) at points `p`."""
    sdf = model(p)
    coords = sdf['model_in']
    values = sdf['model_out'].unsqueeze(0)
    gradient = diff_operators.gradient(values, coords)
    return gradient, diff_operators.divergence(gradient, coords) if not no_curv else None


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


def create_training_data(
        mesh: o3d.t.geometry.TriangleMesh,
        n_on_surf: int,
        on_surf_exceptions: list,
        n_off_surf: int,
        domain_bounds,
        scene: o3d.t.geometry.RaycastingScene,
        no_sdf: bool = False
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

    Returns
    -------
    training_pts: torch.Tensor
    training_normals: torch.Tensor
    training_sdf: torch.Tensor

    See Also
    --------
    sample_on_surface
    """
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


SEED = 271668

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

N_TEST_POINTS = 5000  # Half on surface, half off it. Training points is defined as len(vertices) - TEST_POINTS
BATCH_SIZE = 20000    # Half on surface, half off it.
EPOCHS = 100          # Total steps = EPOCHS * (len(vertices) - N_TEST_POINTS) / BATCH_SIZE
# METHODS = ["rbf", "siren", "i3d", "i3dcurv"]
METHODS = ["i3d", "siren"]
MESH_TYPE = "armadillo"
N_RUNS = 1

netconfig_map = {
    "armadillo": {
        "hidden_layer_config": [256, 256, 256, 256],
        "w0": 60,
        "ww": None,
    },
    "bunny": {
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
    "armadillo": "D:/Users/tiago/projects/high_order_derivative_learning_for_graphics/data/armadillo.ply",
    "bunny": "D:/Users/tiago/projects/high_order_derivative_learning_for_graphics/data/bunny.ply",
}

mesh_data = mesh_map.get(MESH_TYPE, None)
if mesh_data is None:
    raise ValueError(f"Invalid mesh provided \"{MESH_TYPE}\".")

mesh = o3d.io.read_triangle_mesh(mesh_data)
mesh.compute_vertex_normals()
mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
print(mesh)

min_bound = np.array([-1, -1, -1])
max_bound = np.array([1, 1, 1])

# Marching cubes inputs
MC_RESOLUTION = 256
voxel_size = 2.0 / (MC_RESOLUTION - 1)
samples = gen_mc_coordinate_grid(MC_RESOLUTION, voxel_size)

# Create a raycasting scene to perform the SDF querying
scene = o3d.t.geometry.RaycastingScene()
_ = scene.add_triangles(mesh)

# Creating the test set
N = len(mesh.vertex["positions"])

test_surf_pts, test_surf_idx = sample_on_surface(
    mesh,
    N_TEST_POINTS // 2,
)
test_surf_pts = torch.from_numpy(test_surf_pts.numpy())

test_domain_pts = np.random.uniform(
    min_bound, max_bound,
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


if "rbf" in METHODS:
    training_stats = {
        "mean_abs_error": [-1] * N_RUNS,
        "max_abs_error": [-1] * N_RUNS,
        "mean_abs_error_on_surface": [-1] * N_RUNS,
        "max_abs_error_on_surface": [-1] * N_RUNS,
        "mean_abs_error_off_surface": [-1] * N_RUNS,
        "max_abs_error_off_surface": [-1] * N_RUNS,
        "training_times_s": [-1] * N_RUNS
    }
    i = 0
    while i < N_RUNS:
        training_pts, _, training_sdf = create_training_data(
            mesh, BATCH_SIZE // 2, test_surf_idx, BATCH_SIZE // 2,
            [min_bound, max_bound], scene
        )

        start = time.time()
        interp = RBFInterpolator(
            training_pts.detach().numpy(),
            training_sdf.detach().numpy(),
            kernel="cubic",
            neighbors=30
        )
        total_time = time.time() - start

        # Inference on the test data.
        y_rbf = interp(test_pts.detach().numpy())
        errs = torch.abs(test_sdf.detach() - torch.from_numpy(y_rbf))
        errs_on_surf = errs[:test_surf_pts.shape[0]]
        errs_off_surf = errs[test_surf_pts.shape[0]:]
        print(f"RBF Results: MABSE {errs.mean():.3}"
              f" -- MAXERR {errs.max().item():.3} --- Runtime {total_time} s")
        print(f"Errors on surface --- "
              f"MABSE {errs_on_surf.mean():.3} -- MAXERR {errs_on_surf.max().item():.3}")
        print(f"Errors off surface --- "
              f"MABSE {errs_off_surf.mean():.3} -- MAXERR {errs_off_surf.max().item():.3}")

        training_stats["training_times_s"][i] = total_time
        training_stats["max_abs_error"][i] = errs.max().item()
        training_stats["mean_abs_error"][i] = errs.mean().item()
        training_stats["max_abs_error_on_surface"][i] = errs_on_surf.max().item()
        training_stats["mean_abs_error_on_surface"][i] = errs_on_surf.mean().item()
        training_stats["max_abs_error_off_surface"][i] = errs_off_surf.max().item()
        training_stats["mean_abs_error_off_surface"][i] = errs_off_surf.mean().item()

        i += 1

    verts, faces, total_time = run_marching_cubes(samples, interp, MC_RESOLUTION)
    print(f"Marching cubes inference time {total_time:.3} s")
    save_ply(verts, faces, f"mc_rbf_{MESH_TYPE}.ply")
    stats_df = pd.DataFrame.from_dict(training_stats)
    stats_df.to_csv(
        f"stats_rbf_{MESH_TYPE}.csv", sep=";", index=False
    )
    stats_df.mean().to_csv(f"stats_rbf_mean_{MESH_TYPE}.csv", sep=";",
                           index=False)
    stats_df.std().to_csv(f"stats_rbf_stddev_{MESH_TYPE}.csv", sep=";",
                          index=False)

################## SIREN
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
        "training_times_s": [-1] * N_RUNS
    }
    i = 0
    while i < N_RUNS:
        # Model training
        training_loss = {}
        model = SIREN(3, 1, **netconfig).cuda()
        print(model)
        print("# parameters =", parameters_to_vector(model.parameters()).numel())
        optim = torch.optim.Adam(lr=1e-4, params=model.parameters())

        # Start of the training loop
        start_time = time.time()
        for e in range(N_STEPS):
            training_pts, training_normals, training_sdf = \
                create_training_data(
                    mesh,
                    BATCH_SIZE // 2,
                    test_surf_idx,
                    BATCH_SIZE // 2,
                    [min_bound, max_bound],
                    scene,
                    no_sdf=True
                )

            training_pts.cuda()
            training_normals.cuda()
            training_sdf.cuda()

            gt = {
                "sdf": training_sdf.float().unsqueeze(1),
                "normals": training_normals.float(),
            }

            optim.zero_grad()

            y = model(training_pts)
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

            if not e % 100 and e > 0:
                print(f"Epoch {e} --- Loss {running_loss.item()}")

        total_time = time.time() - start_time

        # fig, ax = plt.subplots(1)
        # for k in training_loss:
        #     ax.plot(list(range(N_STEPS)), training_loss[k], label=k)

        # fig.legend()
        # plt.savefig(f"loss_siren_{MESH_TYPE}.png")

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

        training_stats["training_times_s"][i] = total_time
        training_stats["max_abs_error"][i] = errs.max().item()
        training_stats["mean_abs_error"][i] = errs.mean().item()
        training_stats["max_abs_error_on_surface"][i] = errs_on_surf.max().item()
        training_stats["mean_abs_error_on_surface"][i] = errs_on_surf.mean().item()
        training_stats["max_abs_error_off_surface"][i] = errs_off_surf.max().item()
        training_stats["mean_abs_error_off_surface"][i] = errs_off_surf.mean().item()
        training_stats["max_normal_alignment"][i] = errs_normals.max().item()
        training_stats["mean_normal_alignment"][i] = errs_normals.mean().item()

        i += 1

    model.eval()
    verts, faces, total_time = run_marching_cubes(samples, model, MC_RESOLUTION)
    print(f"Marching cubes inference time {total_time:.3} s")
    save_ply(verts, faces, f"mc_siren_{MESH_TYPE}.ply")
    torch.save(model.state_dict(), f"weights_siren_{MESH_TYPE}.pth")
    stats_df = pd.DataFrame.from_dict(training_stats)
    stats_df.to_csv(
        f"stats_siren_{MESH_TYPE}.csv", sep=";", index=False
    )
    stats_df.mean().to_csv(f"stats_siren_mean_{MESH_TYPE}.csv", sep=";",
                           index=False)
    stats_df.std().to_csv(f"stats_siren_stddev_{MESH_TYPE}.csv", sep=";",
                          index=False)

################## I3D
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
        "training_times_s": [-1] * N_RUNS
    }
    i = 0
    while i < N_RUNS:
        # Model training
        training_loss = {}
        model = SIREN(3, 1, **netconfig).cuda()
        print(model)
        print("# parameters =", parameters_to_vector(model.parameters()).numel())
        optim = torch.optim.Adam(lr=1e-4, params=model.parameters())

        # Start of the training loop
        start_time = time.time()
        for e in range(N_STEPS):
            training_pts, training_normals, training_sdf = \
                create_training_data(
                    mesh,
                    BATCH_SIZE // 2,
                    test_surf_idx,
                    BATCH_SIZE // 2,
                    [min_bound, max_bound],
                    scene
                )

            training_pts.cuda()
            training_normals.cuda()
            training_sdf.cuda()

            gt = {
                "sdf": training_sdf.float().unsqueeze(1),
                "normals": training_normals.float(),
            }

            optim.zero_grad()

            y = model(training_pts)
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

            if not e % 100 and e > 0:
                print(f"Epoch {e} --- Loss {running_loss.item()}")

        total_time = time.time() - start_time
        # fig, ax = plt.subplots(1)
        # for k in training_loss:
        #     ax.plot(list(range(N_STEPS)), training_loss[k], label=k)

        # fig.legend()
        # plt.savefig(f"loss_i3d_{MESH_TYPE}.png")

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

        training_stats["training_times_s"][i] = total_time
        training_stats["max_abs_error"][i] = errs.max().item()
        training_stats["mean_abs_error"][i] = errs.mean().item()
        training_stats["max_abs_error_on_surface"][i] = errs_on_surf.max().item()
        training_stats["mean_abs_error_on_surface"][i] = errs_on_surf.mean().item()
        training_stats["max_abs_error_off_surface"][i] = errs_off_surf.max().item()
        training_stats["mean_abs_error_off_surface"][i] = errs_off_surf.mean().item()
        training_stats["max_normal_alignment"][i] = errs_normals.max().item()
        training_stats["mean_normal_alignment"][i] = errs_normals.mean().item()

        i += 1

    model.eval()
    verts, faces, total_time = run_marching_cubes(samples, model, MC_RESOLUTION)
    print(f"Marching cubes inference time {total_time:.3} s")
    save_ply(verts, faces, f"mc_i3d_{MESH_TYPE}.ply")
    torch.save(model.state_dict(), f"weights_i3d_{MESH_TYPE}.pth")
    stats_df = pd.DataFrame.from_dict(training_stats)
    stats_df.to_csv(
        f"stats_i3d_{MESH_TYPE}.csv", sep=";", index=False
    )
    stats_df.mean().to_csv(f"stats_i3d_mean_{MESH_TYPE}.csv", sep=";",
                           index=False)
    stats_df.std().to_csv(f"stats_i3d_stddev_{MESH_TYPE}.csv", sep=";",
                          index=False)
