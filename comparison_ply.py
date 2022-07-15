#!/usr/bin/env python
# coding: utf-8


import time
import open3d as o3d
import open3d.core as o3c
import numpy as np
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
        domain_bounds: tuple[np.array, np.array],
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


np.random.seed(271668)
torch.manual_seed(271668)
torch.cuda.manual_seed(271668)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

N_TEST_POINTS = 5000  # Half on surface, half off it. Training points is defined as len(vertices) - TEST_POINTS
BATCH_SIZE = 20000    # Half on surface, half off it.
EPOCHS = 100          # Total steps = EPOCHS * len(vertices) / BATCH_SIZE
METHODS = ["rbf", "siren", "i3d", "i3dcurv"]
MESH_TYPE = "armadillo"

mesh_map = {
    "armadillo": o3d.data.ArmadilloMesh(),
    "bunny": o3d.data.BunnyMesh(),
}

mesh_data = mesh_map.get(MESH_TYPE, None)
if mesh_data is None:
    raise ValueError(f"Invalid mesh provided \"{MESH_TYPE}\".")

mesh = o3d.io.read_triangle_mesh(mesh_data.path)
mesh.compute_vertex_normals()
mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
print(mesh)

# Normalizing the vertices
min_bound = mesh.vertex['positions'].min(0).numpy()
max_bound = mesh.vertex['positions'].max(0).numpy()

v = (mesh.vertex["positions"] - min_bound) / (max_bound - min_bound)
v = 1.8 * v - 0.9
mesh.vertex["positions"] = v

min_bound = np.array([-1, -1, -1])
max_bound = np.array([1, 1, 1])

# Marching cubes inputs
MC_RESOLUTION = 128
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
    tot_time = time.time() - start

    # Inference on the test data.
    y_rbf = interp(test_pts.detach().numpy())
    errs = torch.abs(test_sdf.detach() - torch.from_numpy(y_rbf))
    errs_on_surf = errs[:test_surf_pts.shape[0]]
    errs_off_surf = errs[test_surf_pts.shape[0]:]
    print(f"RBF Results: MABSE {errs.mean():.3}"
          f" -- MAXERR {errs.max().item():.3} --- Runtime {tot_time} s")
    print(f"Errors on surface --- "
          f"MABSE {errs_on_surf.mean():.3} -- MAXERR {errs_on_surf.max().item():.3}")
    print(f"Errors off surface --- "
          f"MABSE {errs_off_surf.mean():.3} -- MAXERR {errs_off_surf.max().item():.3}")

    verts, faces, total_time = run_marching_cubes(samples, interp, MC_RESOLUTION)
    tot_time = time.time() - start
    print(f"Marching cubes inference time {total_time:.3} s")
    save_ply(verts, faces, f"mc_rbf_{MESH_TYPE}.ply")

################## SIREN
if "siren" in METHODS:
    N_TRAINING_POINTS = N - N_TEST_POINTS
    N_STEPS = round(EPOCHS * ((N - N_TEST_POINTS) / BATCH_SIZE))
    print(f"# of steps: {N_STEPS}")

    # Model training
    training_loss = {}
    model = SIREN(3, 1, [128, 128, 128])
    print(model)
    print("# parameters =", parameters_to_vector(model.parameters()).numel())
    optim = torch.optim.Adam(lr=1e-3, params=model.parameters())

    # Start of the training loop
    for e in range(N_STEPS):
        training_pts, training_normals, training_sdf = create_training_data(
            mesh,
            BATCH_SIZE // 2,
            test_surf_idx,
            BATCH_SIZE // 2,
            [min_bound, max_bound],
            scene,
            no_sdf=True
        )

        gt = {
            "sdf": training_sdf.float().unsqueeze(1),
            "normals": training_normals.float(),
        }

        optim.zero_grad()

        y = model(training_pts)
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

        if not e % 10 and e > 0:
            print(f"Epoch {e} --- Loss {running_loss.item()}")

    model.eval()
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

    model.eval()
    verts, faces, total_time = run_marching_cubes(samples, model, MC_RESOLUTION)
    print(f"Marching cubes inference time {total_time:.3} s")
    save_ply(verts, faces, f"mc_siren_{MESH_TYPE}.ply")
    torch.save(model.state_dict(), f"weights_siren_{MESH_TYPE}.pth")

################## I3D
if "i3d" in METHODS:
    N_TRAINING_POINTS = N - N_TEST_POINTS
    N_STEPS = round(EPOCHS * ((N - N_TEST_POINTS) / BATCH_SIZE))
    print(f"# of steps: {N_STEPS}")

    # Model training
    training_loss = {}
    model = SIREN(3, 1, [128, 128, 128])
    print(model)
    print("# parameters =", parameters_to_vector(model.parameters()).numel())
    optim = torch.optim.Adam(lr=1e-3, params=model.parameters())

    # Start of the training loop
    for e in range(N_STEPS):
        training_pts, training_normals, training_sdf = create_training_data(
            mesh,
            BATCH_SIZE // 2,
            test_surf_idx,
            BATCH_SIZE // 2,
            [min_bound, max_bound],
            scene
        )

        gt = {
            "sdf": training_sdf.float().unsqueeze(1),
            "normals": training_normals.float(),
        }

        optim.zero_grad()

        y = model(training_pts)
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

        if not e % 10 and e > 0:
            print(f"Epoch {e} --- Loss {running_loss.item()}")

    model.eval()
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

    model.eval()
    verts, faces, total_time = run_marching_cubes(samples, model, MC_RESOLUTION)
    print(f"Marching cubes inference time {total_time:.3} s")
    save_ply(verts, faces, f"mc_i3d_{MESH_TYPE}.ply")
    torch.save(model.state_dict(), f"weights_i3d_{MESH_TYPE}.pth")
