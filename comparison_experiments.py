#!/usr/bin/env python
# coding: utf-8


import argparse
import json
import os.path as osp
import numpy as np
from scipy.interpolate import RBFInterpolator
import torch
from torch.nn.utils import parameters_to_vector
from loss_functions import true_sdf as loss_fn
from meshing import (convert_sdf_samples_to_ply, gen_mc_coordinate_grid,
                     save_ply)
from model import SIREN


RADIUS = 0.9
EPOCHS = 250


def sdf_sphere(p: np.ndarray, r: float = 1.0):
    return np.linalg.norm(p) - np.abs(r)


def grad_sdf_sphere(p: np.ndarray):
    if np.linalg.norm(p) < 1e-5:  # Handling points too close to the origin
        return np.zeros(p.shape)
    n = np.linalg.norm(p)
    return p / n


def gen_points_on_sphere(n: int, r: float = 1.0) -> np.ndarray:
    return np.array([
        (p / np.linalg.norm(p)) * r for p in np.random.rand(n, 3)
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparison tests of i3d and other models.")

    parser.add_argument("-n", "--training_points", default=1500, type=int,
                        help="Number of training points to use")
    parser.add_argument("-t", "--test_points", default=1500, type=int,
                        help="Number of test points to use.")
    parser.add_argument("-i", "--mesh", default="sphere", type=str,
                        help="Base mesh to use. May be sphere, torus, dtorus or"
                        " the full path to a PLY file.")
    parser.add_argument("-p", "--fraction_on_surface", default=0.7, type=float,
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

    # Defining points on the surface of a sphere
    sphere_pts = gen_points_on_sphere(
        round(args.training_points * args.fraction_on_surface),
        r=RADIUS
    )

    # Adding random domain samples to the points.
    domain_pts = np.random.rand(
        round(args.training_points * (1 - args.fraction_on_surface)),
        3
    ) * 2 - 1

    training_pts = np.vstack((sphere_pts, domain_pts))

    # Shuffling and calculating the SDF.
    np.random.shuffle(training_pts)
    training_normals = np.array([grad_sdf_sphere(p) for p in training_pts])
    training_sdf = np.array([sdf_sphere(p, r=RADIUS) for p in training_pts])

    test_pts = np.random.rand(args.test_points, 3) * 2 - 1
    test_sdf = np.array([
        sdf_sphere(p) for p in test_pts
    ])

    # Marching cubes inputs
    voxel_size = 2.0 / (args.mc_resolution - 1)
    samples = gen_mc_coordinate_grid(args.mc_resolution, voxel_size)

    if "rbf" in args.methods:
        # Building the interpolant.
        interp = RBFInterpolator(training_pts, training_sdf, kernel="cubic")

        # Inference on the test data.
        y_rbf = interp(test_pts)
        errs = (test_sdf - y_rbf) ** 2
        print(f"RBF Results: SSE {errs.sum():.3} --- MSE {errs.mean():.3}")

        # Marching cubes
        samples_sdf = interp(samples[:, :3]).reshape([args.mc_resolution] * 3)
        verts, faces, _, _ = convert_sdf_samples_to_ply(
            samples_sdf, [-1, -1, -1], voxel_size, None, None
        )
        save_ply(verts, faces, "mc_rbf.ply")

    if "i3d" in args.methods:
        netconfig = {
            "hidden_layer_config": [32, 32, 32],
            "w0": 5,
            "ww": None
        }
        if args.i3d_config is not None and osp.exists(path := osp.expand_user(args.i3d_config)):
            with open(path, "r") as jin:
                contents = json.load(jin)
                netconfig.update(contents["network"])
                EPOCHS = contents["num_epochs"]

        model = SIREN(3, 1, **netconfig)
        print(model)
        print("# parameters =", parameters_to_vector(model.parameters()).numel())
        optim = torch.optim.Adam(lr=1e-3, params=model.parameters())

        # Training the model
        model.train()
        for e in range(EPOCHS):
            # Defining points on the surface of a sphere
            sphere_pts = torch.from_numpy(gen_points_on_sphere(
                round(args.training_points * args.fraction_on_surface),
                r=RADIUS
            ))

            # Adding random domain samples to the points.
            domain_pts = torch.rand(
                round(args.training_points * (1 - args.fraction_on_surface)),
                3
            ) * 2 - 1

            training_pts = torch.row_stack((sphere_pts, domain_pts)).float()

            # Shuffling and calculating the SDF.
            # np.random.shuffle(training_pts)
            training_normals = torch.Tensor([grad_sdf_sphere(p) for p in training_pts.numpy()])
            training_sdf = torch.Tensor([sdf_sphere(p, r=RADIUS) for p in training_pts.numpy()])

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

            print(f"Epoch {e} --- Loss: {training_loss.item()}")

            training_loss.backward()
            optim.step()

        # Testing the model
        # Inference on the test data.
        test_pts_t = torch.from_numpy(test_pts).float()
        test_sdf_t = torch.from_numpy(test_sdf).float().unsqueeze(1)
        model.eval()
        with torch.no_grad():
            y_i3d = model(test_pts_t)["model_out"]
            errs = (test_sdf_t - y_i3d) ** 2
            print(f"i3d Results: SSE {errs.sum():.3} --- MSE {errs.mean():.3}")

            samples_sdf = model(samples[:, :3])["model_out"].reshape([args.mc_resolution] * 3)
            verts, faces, _, _ = convert_sdf_samples_to_ply(
                samples_sdf, [-1, -1, -1], voxel_size, None, None
            )
            save_ply(verts, faces, "mc_i3d.ply")
