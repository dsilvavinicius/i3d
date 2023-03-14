#!/usr/bin/env python
# coding: utf-8

"""
Experiments with calculating the SDF for a batch of points and reusing it for N
iterations.
"""

import argparse
import copy
import math
import os
import os.path as osp
import shutil
import time
import yaml
import numpy as np
import open3d as o3d
import open3d.core as o3c
from plyfile import PlyData
import torch
from torch.nn.utils import parameters_to_vector
from loss_functions import true_sdf
from model import SIREN


def curvature_segmentation(
    vertices: torch.Tensor,
    n_samples: int,
    bin_edges: np.array,
    proportions: np.array,
    device: str
):
    """Samples `n_samples` from `vertices` based on their curvatures. Each
    sample is a row of `vertices`.

    Parameters
    ----------
    vertices: torch.Tensor
        The vertices to sample. Note that each row is a vertex and the
        curvatures must be stored in the second-to-last column of `vertices`.

    n_samples: int
        The number of items to fetch from `vertices`.

    bin_edges: torch.Tensor
        The [minimum, low-medium threshold, medium-high threshold, maximum]
        curvature values in `vertices`. These values define thresholds between
        low and medium curvature values, and medium to high curvatures.

    proportions: torch.Tensor
        The percentage of points to fetch for each curvature band per batch of
        `n_samples`.

    Returns
    -------
    samples: torch.Tensor
        The sampled vertices.
    """
    curvatures = vertices[..., -2]

    # TODO: Da pra fazer na inicialização
    low_curvature_pts = vertices[curvatures < bin_edges[1], ...]
    med_curvature_pts = vertices[(curvatures >= bin_edges[1]) & (curvatures < bin_edges[2]), ...]
    high_curvature_pts = vertices[curvatures >= bin_edges[2], ...]

    n_low_curvature = int(math.floor(proportions[0] * n_samples))
    n_med_curvature = int(math.ceil(proportions[1] * n_samples))
    n_high_curvature = n_samples - (n_low_curvature + n_med_curvature)

    low_idx = torch.randperm(low_curvature_pts.shape[0], device=device)[:n_low_curvature]
    med_idx = torch.randperm(med_curvature_pts.shape[0], device=device)[:n_med_curvature]
    high_idx = torch.randperm(high_curvature_pts.shape[0], device=device)[:n_high_curvature]

    ret = torch.row_stack((
        low_curvature_pts[low_idx, ...],
        med_curvature_pts[med_idx, ...],
        high_curvature_pts[high_idx, ...]
    ))

    return ret


def sample_on_surface(
        vertices: torch.Tensor,
        n_points: int,
        device: str
):
    """Samples points in a torch tensor

    Parameters
    ----------
    vertices: torch.Tensor
        A mode-2 tensor where each row is a vertex.

    n_points: int
        The number of points to sample. If `n_points` == `vertices.shape[0]`,
        we simply return `vertices` without any change.

    device: str or torch.device
        The device where we should generate the indices of sampled points.

    Returns
    -------
    sampled: torch.tensor
        The points sampled from `vertices`. If
        `n_points` == `vertices.shape[0]`, then we simply return `vertices`.

    idx: torch.tensor
        The indices of points sampled from `vertices`. Naturally, these are
        row indices in `vertices`.

    See Also
    --------
    torch.randperm
    """
    if n_points == vertices.shape[0]:
        return vertices, torch.arange(end=n_points, step=1, device=device)
    idx = torch.randperm(vertices.shape[0], device=device)[:n_points]
    sampled = vertices[idx, ...]
    return sampled, idx


def create_training_data(
    vertices: torch.tensor,
    n_on_surf: int,
    n_off_surf: int,
    domain_bounds: list,
    scene: o3d.t.geometry.RaycastingScene,
    device: torch.device = torch.device("cpu"),
    no_sdf: bool = False,
    use_curvature: bool = False,
    curvature_fracs: list = [],
    curvature_threshs: list = [],
):
    """Creates a set of training data with coordinates, normals and SDF
    values.

    Parameters
    ----------
    vertices: torch.tensor
        A mode-2 tensor with the mesh vertices.

    n_on_surf: int
        # of points to sample from the mesh.

    n_off_surf: int
        # of points to sample from the domain. Note that we sample points
        uniformely at random from the domain.

    domain_bounds: tuple[np.array, np.array]
        Bounds to use when sampling points from the domain.

    scene: o3d.t.geometry.RaycastingScene
        Open3D raycasting scene to use when querying SDF for domain points.

    device: str or torch.device, optional
        The compute device where `vertices` is stored. By default its
        torch.device("cpu")

    no_sdf: boolean, optional
        Don't query SDF for domain points, instead we mark them with SDF = -1.

    use_curvature: boolean, optional
        Indicates if we must use the curvature to perform sampling on surface
        points. Note that we expect the curvature to be the second-to-last
        column in `vertices`.

    curvature_fracs: list, optional
        The fractions of points to sample per curvature band. Only used when
        `use_curvature` is True.

    curvature_threshs: list
        The curvature values to use when defining low, medium and high
        curvatures. Only used when `use_curvature` is True.

    Returns
    -------
    coords: dict[str => list[torch.Tensor]]
        A dictionary with points sampled from the surface (key = "on_surf")
        and the domain (key = "off_surf"). Each dictionary element is a list
        of tensors with the vertex coordinates as the first element of said
        list, the normals as the second element, finally, the SDF is the last
        element.

    See Also
    --------
    sample_on_surface, curvature_segmentation
    """
    if use_curvature:
        surf_pts = curvature_segmentation(
            vertices, n_on_surf, curvature_threshs, curvature_fracs,
            device=device
        )
    else:
        surf_pts, _ = sample_on_surface(
            vertices,
            n_on_surf,
            device=device
        )

    coord_dict = {
        "on_surf": [surf_pts[..., :3],
                    surf_pts[..., 3:6],
                    surf_pts[..., -1]]
    }

    if n_off_surf != 0:
        domain_pts = np.random.uniform(
            domain_bounds[0], domain_bounds[1],
            (n_off_surf, 3)
        )

        if no_sdf is False:
            domain_pts = o3c.Tensor(domain_pts, dtype=o3c.Dtype.Float32)
            domain_sdf = scene.compute_signed_distance(domain_pts)
            domain_sdf = torch.from_numpy(domain_sdf.numpy()).to(device)
        else:
            domain_sdf = torch.full(
                (n_off_surf, 1),
                fill_value=-1,
                device=device
            )

        domain_pts = torch.from_numpy(domain_pts.numpy()).to(device)
        coord_dict["off_surf"] = [domain_pts,
                                  torch.zeros_like(domain_pts, device=device),
                                  domain_sdf]
    return coord_dict


def read_ply(
    path: str,
    with_curvatures: bool = False
):
    """Reads a PLY file with position, normal and, optionally curvature data.

    Note that we expect the input ply to contain x,y,z vertex data, as well
    as nx,ny,nz normal data and the curvature stored in the `quality` vertex
    property, if `with_curvatures` is `True`, else we don't need the `quality`
    attribute set.

    Parameters
    ----------
    path: str, PathLike
        Path to the ply file. We except the file to be in binary format.

    with_curvatures: boolean, optional
        Whether the PLY file has curvatures (and we should read them) or not.
        Default value is False

    Returns
    -------
    mesh: o3d.t.geometry.TriangleMesh
        The fully constructed Open3D Triangle Mesh. By default, the mesh is
        allocated on the CPU:0 device.

    vertices: torch.tensor
        The same vertex information as stored in `mesh`, augmented by the SDF
        values as the last column (a column of zeroes). Returned for easier,
        structured access.

    See Also
    --------
    PlyData.read, o3d.t.geometry.TriangleMesh
    """
    # Reading the PLY file with curvature info
    n_columns = 7  # x, y, z, nx, ny, nz, sdf
    if with_curvatures:
        n_columns = 8  # x, y, z, nx, ny, nz, curvature, sdf
    with open(path, "rb") as f:
        plydata = PlyData.read(f)
        num_verts = plydata["vertex"].count
        vertices = np.zeros(shape=(num_verts, n_columns), dtype=np.float32)
        vertices[:, 0] = plydata["vertex"].data["x"]
        vertices[:, 1] = plydata["vertex"].data["y"]
        vertices[:, 2] = plydata["vertex"].data["z"]
        vertices[:, 3] = plydata["vertex"].data["nx"]
        vertices[:, 4] = plydata["vertex"].data["ny"]
        vertices[:, 5] = plydata["vertex"].data["nz"]
        if with_curvatures:
            vertices[:, 6] = np.abs(plydata["vertex"].data["quality"])

        faces = np.stack(plydata["face"].data["vertex_indices"])

    # Converting the PLY data to open3d format
    device = o3c.Device("CPU:0")
    mesh = o3d.t.geometry.TriangleMesh(device)
    mesh.vertex["positions"] = o3c.Tensor(vertices[:, :3], dtype=o3c.float32)
    mesh.vertex["normals"] = o3c.Tensor(vertices[:, 3:6], dtype=o3c.float32)
    if with_curvatures:
        mesh.vertex["curvatures"] = o3c.Tensor(vertices[:, -2],
                                               dtype=o3c.float32)
    mesh.triangle["indices"] = o3c.Tensor(faces, dtype=o3c.int32)

    return mesh, torch.from_numpy(vertices).requires_grad_(False)


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(
        description="Experiments with SDF querying at regular intervals."
    )
    parser.add_argument(
        "meshpath",
        help="Path to the mesh to use for training. We only handle PLY files."
    )
    parser.add_argument(
        "outputpath",
        help="Path to the output folder (will be created if necessary)."
    )
    parser.add_argument(
        "configpath",
        help="Path to the configuration file with the network's description."
    )
    parser.add_argument(
        "--device", "-d", type=str, default="cuda:0",
        help="The device to perform the training on. Uses CUDA:0 by default."
    )
    parser.add_argument(
        "--nepochs", "-n", type=int, default=0,
        help="Number of training epochs for each mesh."
    )
    parser.add_argument(
        "--omega0", "-o", type=int, default=0,
        help="SIREN Omega 0 parameter."
    )
    parser.add_argument(
        "--omegaW", "-w", type=int, default=0,
        help="SIREN Omega 0 parameter for hidden layers."
    )
    args = parser.parse_args()

    if not osp.exists(args.configpath):
        raise FileNotFoundError(
            f"Experiment configuration file \"{args.configpath}\" not found."
        )

    if not osp.exists(args.meshpath):
        raise FileNotFoundError(
            f"Mesh file \"{args.meshpath}\" not found."
        )

    with open(args.configpath, "r") as fin:
        config = yaml.safe_load(fin)

    print(f"Saving results in {args.outputpath}")
    if not osp.exists(args.outputpath):
        os.makedirs(args.outputpath)

    shutil.copy(args.configpath, osp.join(args.outputpath, "config.yaml"))

    trainingcfg = config["training"]
    SEED = 668123
    EPOCHS = trainingcfg["epochs"] if not args.nepochs else args.nepochs
    BATCH = trainingcfg["batchsize"]
    REFRESH_SDF_AT_PERC_STEPS = trainingcfg["resample_sdf_at"] / EPOCHS

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    devstr = args.device
    if "cuda" in devstr and not torch.cuda.is_available():
        devstr = "cpu"
        print("No CUDA available devices found on system. Using CPU.")

    device = torch.device(devstr)

    samplingcfg = config.get("sampling", None)
    withcurvature = samplingcfg is not None and samplingcfg["type"] == "curvature"

    mesh, vertices = read_ply(args.meshpath, with_curvatures=withcurvature)
    vertices = vertices.to(device)
    N = vertices.shape[0]
    nsteps = round(EPOCHS * (2 * N / BATCH))
    warmup_steps = nsteps // 10
    refresh_sdf_nsteps = max(1, round(REFRESH_SDF_AT_PERC_STEPS * nsteps))
    print(f"Refresh SDF at every {refresh_sdf_nsteps} training steps")
    print(f"Total # of training steps = {nsteps}")

    min_bound = np.array([-1, -1, -1])
    max_bound = np.array([1, 1, 1])

    # Create a raycasting scene to perform the SDF querying
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)

    # Create the model and optimizer
    netcfg = config["network"]
    model = SIREN(
        netcfg["in_coords"],
        netcfg["out_coords"],
        hidden_layer_config=netcfg["hidden_layers"],
        w0=netcfg["omega_0"] if not args.omega0 else args.omega0,
        ww=netcfg["omega_w"] if not args.omegaW else args.omegaW
    ).to(device)
    print(model)
    print("# parameters =", parameters_to_vector(model.parameters()).numel())
    optim = torch.optim.Adam(lr=1e-4, params=model.parameters())
    training_loss = {}

    off_surf_samples = None

    # Binning the curvatures
    curvature_fracs = []
    curvature_threshs = []
    if withcurvature:
        l1, l2 = torch.quantile(
            vertices[..., -2],
            torch.tensor(samplingcfg["curvature_percentiles"], device=device),
            dim=0
        )
        curvature_threshs = [
            torch.min(vertices[..., -2]),
            l1,
            l2,
            torch.max(vertices[..., -2])
        ]
        curvature_fracs = samplingcfg["curvature_fractions"]

    # Training loop
    trainingpts = torch.zeros((BATCH, 3), device=device)
    trainingnormals = torch.zeros((BATCH, 3), device=device)
    trainingsdf = torch.zeros((BATCH), device=device)
    nonsurf = round(BATCH * 0.5)
    noffsurf = round(BATCH * 0.5)

    best_loss = torch.inf
    best_weights = None

    start_training_time = time.time()
    for step in range(nsteps):
        if not step % refresh_sdf_nsteps:
            # We will recalculate the SDF points at this # of steps
            samples = create_training_data(
                vertices,
                n_on_surf=nonsurf,
                n_off_surf=noffsurf,
                domain_bounds=[min_bound, max_bound],
                scene=scene,
                device=device,
                use_curvature=withcurvature,
                curvature_fracs=curvature_fracs,
                curvature_threshs=curvature_threshs
            )
            off_surf_samples = samples["off_surf"]
            off_surf_samples = [v.to(device) for v in off_surf_samples]

            trainingpts[noffsurf:, ...] = off_surf_samples[0]
            trainingnormals[noffsurf:, ...] = off_surf_samples[1]
            trainingsdf[noffsurf:, ...] = off_surf_samples[2]
        else:
            samples = create_training_data(
                vertices,
                n_on_surf=nonsurf,
                n_off_surf=0,
                domain_bounds=[min_bound, max_bound],
                scene=scene,
                device=device,
                use_curvature=withcurvature,
                curvature_fracs=curvature_fracs,
                curvature_threshs=curvature_threshs
            )

        trainingpts[:nonsurf, ...] = samples["on_surf"][0]
        trainingnormals[:nonsurf, ...] = samples["on_surf"][1]
        trainingsdf[:nonsurf] = samples["on_surf"][2]

        gt = {
            "sdf": trainingsdf.float().unsqueeze(1),
            "normals": trainingnormals.float(),
        }

        optim.zero_grad(set_to_none=True)
        y = model(trainingpts)
        loss = true_sdf(y, gt)

        running_loss = torch.zeros((1, 1), device=device)
        for k, v in loss.items():
            running_loss += v
            if k not in training_loss:
                training_loss[k] = [v.detach()]
            else:
                training_loss[k].append(v.detach())

        running_loss.backward()
        optim.step()

        if step > warmup_steps and running_loss.item() < best_loss:
            best_weights = copy.deepcopy(model.state_dict())
            best_loss = running_loss.item()

        if not step % 100 and step > 0:
            print(f"Step {step} --- Loss {running_loss.item()}")

    training_time = time.time() - start_training_time
    print(f"training took {training_time} s")
    torch.save(
        model.state_dict(), osp.join(args.outputpath, "weights.pth")
    )
    torch.save(
        best_weights, osp.join(args.outputpath, "best.pth")
    )
