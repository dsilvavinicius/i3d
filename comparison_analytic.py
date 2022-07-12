#!/usr/bin/env python
# coding: utf-8


import argparse
import json
import time
import os.path as osp
import numpy as np
from scipy.interpolate import RBFInterpolator
import torch
from torch.nn.utils import parameters_to_vector
import diff_operators
from loss_functions import true_sdf as loss_fn
from meshing import (convert_sdf_samples_to_ply, gen_mc_coordinate_grid,
                     save_ply)
from model import SIREN


class sdf_sphere(torch.nn.Module):
    def __init__(self, r: float = 1.0):
        super().__init__()
        self.r = torch.Tensor([r])

    def forward(self, p: torch.Tensor) -> dict:
        try:
            p.requires_grad = True
        except:
            pass

        return {
            "model_in": p,
            "model_out": torch.linalg.norm(p, dim=1) - torch.abs(self.r)
        }


class sdf_torus(torch.nn.Module):
    def __init__(self, c:float = 0.6, a: float = 0.5):
        super().__init__()
        self.c = c
        self.a = a

    def forward(self, p: torch.Tensor) -> dict:
        try:
            p.requires_grad = True
        except:
            pass

        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]

        qx = torch.sqrt(x ** 2 + y ** 2) - self.c
        dist = torch.sqrt(qx ** 2 + z ** 2) - self.a ** 2

        return {"model_in": p, "model_out": dist}


class sdf_octahedron(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p):
        p.requires_grad = True

        #f(x,y,z)=x**4 + y**4 + z**4 + 6*x**2*y**2 + 6*y**2*z**2 + 6*z**2*x**2 − 1 
        dist = p[..., 0] ** 4 + p[..., 1] ** 4 + p[..., 2] ** 4 \
            + 6 * p[..., 0] ** 2 * p[..., 1] ** 2 \
            + 6 * p[..., 1] ** 2 * p[..., 2] ** 2 \
            + 6 * p[..., 2] ** 2 * p[..., 0] ** 2 - 1

        return {"model_in": p, "model_out": dist.unsqueeze(-1)}



model_map = {
    "sphere": sdf_sphere(0.9),
    "torus": sdf_torus(0.6, 0.5),
}

netconfig_map = {
    "sphere": {
        "hidden_layer_config": [32, 32, 32],
        "w0": 5,
        "ww": None,
    },
    "torus": {
        "hidden_layer_config": [64, 64, 64],
        "w0": 10,
        "ww": None,
    },
    "default": {
        "hidden_layer_config": [256, 256, 256],
        "w0": 30,
        "ww": None,
    }
}


def grad_sdf(p: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    """Evaluates the gradient of F (`model`) at points `p`."""
    sdf = model(p)
    coords = sdf['model_in']
    values = sdf['model_out'].unsqueeze(0)
    gradient = diff_operators.gradient(values, coords)
    return gradient


def gen_points_on_surf(n: int, model: torch.nn.Module) -> torch.Tensor:
    """Generates `n` points on the surface of F, defined by `model`."""
    P = torch.rand(n, 3) * 2 - 1
    sdf = model(P)

    coords = sdf['model_in']
    values = sdf['model_out'].unsqueeze(0)
    gradient = diff_operators.gradient(values, coords)

    proj = P - gradient * values.squeeze().unsqueeze(1)
    return proj, P


def viz_mesh(mesh_type_path: str):
    """Visualizes the mesh points. For debugging purposes mostly."""
    if mesh_type_path is model_map:
        model = model_map[mesh_type_path]
    else:
        if not osp.exists(osp.expanduser(mesh_type_path)):
            raise FileNotFoundError
        return 1

    proj, P = gen_points_on_surf(1500, model)
    cloud = pyrender.Mesh.from_points(proj.detach(), colors=torch.zeros_like(P))
    domain_colors = torch.Tensor([255, 0, 0] * P.shape[0]).reshape(-1, 3)
    cloud_domain = pyrender.Mesh.from_points(P.detach(), colors=domain_colors)
    light = pyrender.PointLight(intensity=500)
    scene = pyrender.Scene()
    scene.add(light)
    scene.add(cloud)
    scene.add(cloud_domain)
    viewer = pyrender.Viewer(scene, point_size=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comparison tests of i3d and other models. Only for"
        " analytical models. For arbitrary meshes, see \"comparison_ply.py\"."
    )

    parser.add_argument("-n", "--training_points", default=5000, type=int,
                        help="Number of training points to use")
    parser.add_argument("-t", "--test_points", default=5000, type=int,
                        help="Number of test points to use.")
    parser.add_argument("-i", "--input", default="sphere", type=str,
                        help="Base mesh to use. May be sphere, torus, ...")
    parser.add_argument("-p", "--fraction_on_surface", default=0.5, type=float,
                        help="Fraction of points to sample on surface of object "
                        " (for training purposes only). The remaining points"
                        " will be randomly sampled from the domain.")
    parser.add_argument("-c", "--i3d_config", default=None, type=str,
                        help="Path to the i3d experiment JSON file. If not"
                        " provided, a default configuration will be used.")
    parser.add_argument("-m", "--methods", default=["rbf", "i3d"], nargs="*",
                        help="Models to test. Options are: rbf, i3d")
    parser.add_argument("-s", "--seed", default=271668, type=int,
                        help="RNG seed.")
    parser.add_argument("-r", "--mc_resolution", default=64, type=int,
                        help="Marching cubes resolution. If set to 0, will not"
                        " attempt to run it.")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.input in model_map:
        mesh_model = model_map[args.input]
    else:
        raise ValueError("Invalid model given. Must be one of:"
                         f" {list(model_map.keys())}")

    # Creating a test set for error metric evaluation.
    test_domain_pts = torch.rand(args.test_points, 3) * 2 - 1
    test_surf_pts, _ = gen_points_on_surf(args.test_points, mesh_model)
    test_pts = torch.row_stack((test_surf_pts, test_domain_pts))
    with torch.no_grad():
        test_sdf = mesh_model(test_pts)["model_out"]

    # Marching cubes inputs
    voxel_size = 2.0 / (args.mc_resolution - 1)
    samples = gen_mc_coordinate_grid(args.mc_resolution, voxel_size)

    if "rbf" in args.methods:
        surf_pts, _ = gen_points_on_surf(
            round(args.training_points * args.fraction_on_surface),
            mesh_model
        )
        domain_pts = torch.rand(
            round(args.training_points * (1 - args.fraction_on_surface)), 3
        ) * 2 - 1
        training_pts = torch.row_stack((surf_pts, domain_pts)).float()

        training_normals = grad_sdf(training_pts, mesh_model)
        with torch.no_grad():
            training_sdf = mesh_model(training_pts)["model_out"]

        # Building the interpolant.
        start = time.time()
        interp = RBFInterpolator(
            training_pts.detach().numpy(),
            training_sdf.detach().numpy(),
            kernel="cubic"
        )
        tot_time = time.time() - start

        # Inference on the test data.
        y_rbf = interp(test_pts.detach().numpy())
        errs = torch.abs(test_sdf.detach() - torch.from_numpy(y_rbf))
        print(f"RBF Results: SABSE {errs.sum():.3} --- MABSE {errs.mean():.3}"
              f" -- MAXERR {errs.max().item():.3} --- Runtime {tot_time} s")

        # Marching cubes
        samples_sdf = interp(samples[:, :3]).reshape([args.mc_resolution] * 3)
        verts, faces, _, _ = convert_sdf_samples_to_ply(
            samples_sdf, [-1, -1, -1], voxel_size, None, None
        )
        save_ply(verts, faces, f"mc_rbf_{model_name}.ply")

    if "i3d" in args.methods:
        netconfig = netconfig_map.get(args.input, netconfig_map["default"])
        if args.i3d_config is not None and \
           osp.exists(path := osp.expanduser(args.i3d_config)):
            with open(path, "r") as jin:
                contents = json.load(jin)
                netconfig.update(contents["network"])
                EPOCHS = contents["num_epochs"]

        model = SIREN(3, 1, **netconfig)
        # print(model)
        # print("# parameters =", parameters_to_vector(model.parameters()).numel())
        optim = torch.optim.Adam(lr=1e-3, params=model.parameters())

        # Training the model
        model.train()
        start = time.time()
        for e in range(EPOCHS):
            # Adding random domain samples to the points.
            domain_pts = torch.rand(
                round(args.training_points * (1 - args.fraction_on_surface)),
                3
            ) * 2 - 1
            surf_pts, _ = gen_points_on_surf(
                round(args.training_points * args.fraction_on_surface),
                mesh_model
            )
            training_pts = torch.row_stack((surf_pts, domain_pts)).float()
            training_normals = grad_sdf(training_pts, mesh_model)
            with torch.no_grad():
                training_sdf = mesh_model(training_pts)["model_out"]

            gt = {
                "sdf": training_sdf.float().unsqueeze(1),
                "normals": training_normals.float(),
            }

            optim.zero_grad()

            y = model(training_pts)
            loss = loss_fn(y, gt)

            training_loss = torch.zeros((1, 1))
            for v in loss.values():
                training_loss += v

            # if not e % 10 and e > 0:
            #     print(f"Epoch {e} --- Loss: {training_loss.item()}")

            training_loss.backward()
            optim.step()

        tot_time = time.time() - start

        # Testing the model
        # Inference on the test data.
        if isinstance(test_pts, np.ndarray):
            test_pts = torch.from_numpy(test_pts).float()

        if isinstance(test_pts, np.ndarray):
            test_sdf = torch.from_numpy(test_sdf).float().unsqueeze(1)

        model.eval()
        with torch.no_grad():
            y_i3d = model(test_pts)["model_out"]
            errs = torch.abs(test_sdf - y_i3d)
            print(f"i3d Results: SABSE {errs.sum():.3} ---"
                  f" MABSE {errs.mean():.3} -- MAXERR {errs.max().item():.3}"
                  f" --- Runtime {tot_time:} s")

            samples_sdf = model(samples[:, :3])["model_out"].reshape([args.mc_resolution] * 3)
            verts, faces, _, _ = convert_sdf_samples_to_ply(
                samples_sdf, [-1, -1, -1], voxel_size, None, None
            )
            save_ply(verts, faces, f"mc_i3d_{model_name}.ply")
