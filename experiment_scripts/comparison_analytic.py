#!/usr/bin/env python
# coding: utf-8


import argparse
import json
import time
import os.path as osp
import numdifftools as nd
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector
import diff_operators
from loss_functions import true_sdf, true_sdf_curvature, sdf_sitzmann
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


# From https://stackoverflow.com/questions/20708038/scipy-misc-derivative-for-multiple-argument-function
def partial_function(f___,input,pos,value):
    tmp  = input[pos]
    input[pos] = value
    ret = f___(input[None, ...])
    input[pos] = tmp
    return ret


def partial_derivative(f, point):
    ret = np.empty(len(point))
    for i in range(len(point)):
        fg = lambda x:partial_function(f, point, i, x)
        ret[i] = nd.Derivative(fg, step=1e-6, order=3)(point[i])
    return ret


model_map = {
    "sphere": sdf_sphere(0.9),
    "torus": sdf_torus(0.6, 0.5),
}

netconfig_map = {
    "sphere": {
        "hidden_layer_config": [80, 80],
        "w0": 20,
        "ww": None,
    },
    "torus": {
        "hidden_layer_config": [80, 80],
        "w0": 20,
        "ww": None,
    },
    "default": {
        "hidden_layer_config": [256, 256, 256],
        "w0": 30,
        "ww": None,
    }
}

EPOCHS = 500


def grad_sdf(p: torch.Tensor, model: torch.nn.Module,
             no_curv: bool = False) -> (torch.Tensor, torch.Tensor):
    """Evaluates the gradient of F (`model`) at points `p`."""
    sdf = model(p)
    coords = sdf['model_in']
    values = sdf['model_out'].unsqueeze(0)
    gradient = diff_operators.gradient(values, coords)
    return gradient, diff_operators.divergence(gradient, coords) if not no_curv else None


def gen_points_on_surf(n: int, model: torch.nn.Module) -> (torch.Tensor, torch.Tensor):
    """Generates `n` points on the surface of F, defined by `model`."""
    P = torch.rand(n, 3) * 2 - 1
    sdf = model(P)

    coords = sdf['model_in']
    values = sdf['model_out'].unsqueeze(0)
    gradient = diff_operators.gradient(values, coords)

    proj = P - gradient * values.squeeze().unsqueeze(1)
    return proj, P


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comparison tests of i3d and other models. Only for"
        " analytical models. For arbitrary meshes, see \"comparison_ply.py\"."
    )

    parser.add_argument("--training_points", default=5000, type=int,
                        help="Number of training points to use")
    parser.add_argument("--test_points", default=5000, type=int,
                        help="Number of test points to use.")
    parser.add_argument("-i", "--input", default="sphere", type=str,
                        help="Base mesh for training."
                        " May be \"sphere\" and \"torus\"")
    parser.add_argument("-p", "--fraction_on_surface", default=0.5, type=float,
                        help="Fraction of points to sample on surface of"
                        " object (for training purposes only). The remaining"
                        " points will be randomly sampled from the domain.")
    parser.add_argument("-c", "--i3d_config", default=None, type=str,
                        help="Path to the i3d experiment JSON file. If not"
                        " provided, a default configuration will be used.")
    parser.add_argument("-m", "--methods",
                        default=["rbf", "siren", "i3d", "i3dcurv"],
                        nargs="*",
                        help="Models to test. Options are: \"rbf\", \"siren\", "
                        "\"i3d\", \"i3dcurv\"")
    parser.add_argument("-s", "--seed", default=271668, type=int,
                        help="RNG seed.")
    parser.add_argument("-r", "--mc_resolution", default=64, type=int,
                        help="Marching cubes resolution. If set to 0, will not"
                        " attempt to run it.")
    parser.add_argument("-n", "--num_runs", default=1, type=int,
                        help="Number of times to run the tests. Useful for"
                        " statistics.")
    parser.add_argument("--save_weights", action="store_true", default=False,
                        help="Indicates if we should save the weights.")

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
    test_domain_pts = torch.rand(args.test_points // 2, 3) * 2 - 1
    test_surf_pts, _ = gen_points_on_surf(args.test_points // 2, mesh_model)
    test_pts = torch.row_stack((test_surf_pts, test_domain_pts))
    test_normals, test_curvatures = grad_sdf(test_surf_pts, mesh_model)
    with torch.no_grad():
        test_sdf = mesh_model(test_pts)["model_out"]
        test_sdf[:test_domain_pts.shape[0]] = 0

    # Marching cubes inputs
    voxel_size = 2.0 / (args.mc_resolution - 1)
    samples = gen_mc_coordinate_grid(args.mc_resolution, voxel_size)

    if "rbf" in args.methods:
        training_stats = {
            "mean_abs_error": [-1] * args.num_runs,
            "max_abs_error": [-1] * args.num_runs,
            "mean_abs_error_on_surface": [-1] * args.num_runs,
            "max_abs_error_on_surface": [-1] * args.num_runs,
            "mean_abs_error_off_surface": [-1] * args.num_runs,
            "max_abs_error_off_surface": [-1] * args.num_runs,
            "mean_normal_alignment": [-1] * args.num_runs,
            "max_normal_alignment": [-1] * args.num_runs,
            "execution_times": [-1] * args.num_runs
        }
        i = 0
        while i < args.num_runs:
            surf_pts, _ = gen_points_on_surf(
                round(args.training_points * args.fraction_on_surface),
                mesh_model
            )
            domain_pts = torch.rand(
                round(args.training_points * (1 - args.fraction_on_surface)), 3
            ) * 2 - 1
            training_pts = torch.row_stack((surf_pts, domain_pts)).float()

            training_normals, _ = grad_sdf(training_pts, mesh_model)
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
            n_rbf = torch.Tensor(
                [partial_derivative(interp, test_pts.detach().numpy()[j, :]) for j in range(args.test_points)]
            )
            errs = torch.abs(test_sdf.detach() - torch.from_numpy(y_rbf))
            errs_on_surf = errs[:test_surf_pts.shape[0]]
            errs_off_surf = errs[test_surf_pts.shape[0]:]
            errs_normals = 1 - F.cosine_similarity(
                test_normals[:test_surf_pts.shape[0], ...],
                n_rbf[:test_surf_pts.shape[0], ...],
                dim=-1
            )
            print(f"RBF Results: MABSE {errs.mean():.3}"
                  f" -- MAXERR {errs.max().item():.3} --- Runtime {tot_time} s")
            print(f"Errors on surface --- "
                  f"MABSE {errs_on_surf.mean():.3} -- MAXERR {errs_on_surf.max().item():.3}")
            print(f"Errors off surface --- "
                  f"MABSE {errs_off_surf.mean():.3} -- MAXERR {errs_off_surf.max().item():.3}")
            print(f"Normal alignment errors --- MEAN {errs_normals.mean():.3}"
                  f" --- MAX {errs_normals.max().item():.3}")
            training_stats["execution_times"][i] = tot_time
            training_stats["max_abs_error"][i] = errs.max().item()
            training_stats["mean_abs_error"][i] = errs.mean().item()
            training_stats["max_abs_error_on_surface"][i] = errs_on_surf.max().item()
            training_stats["mean_abs_error_on_surface"][i] = errs_on_surf.mean().item()
            training_stats["max_abs_error_off_surface"][i] = errs_off_surf.max().item()
            training_stats["mean_abs_error_off_surface"][i] = errs_off_surf.mean().item()
            training_stats["max_normal_alignment"][i] = errs_normals.max().item()
            training_stats["mean_normal_alignment"][i] = errs_normals.mean().item()
            i += 1

        # Marching cubes
        samples_sdf = interp(samples[:, :3]).reshape([args.mc_resolution] * 3)
        verts, faces, _, _ = convert_sdf_samples_to_ply(
            samples_sdf, [-1, -1, -1], voxel_size, None, None
        )
        save_ply(verts, faces, f"mc_rbf_{args.input}.ply")
        pd.DataFrame.from_dict(training_stats).to_csv(
            f"stats_rbf_{args.input}.csv", sep=";", index=False
        )

    if "siren" in args.methods:
        netconfig = netconfig_map.get(args.input, netconfig_map["default"])
        if args.i3d_config is not None and \
           osp.exists(path := osp.expanduser(args.i3d_config)):
            with open(path, "r") as jin:
                contents = json.load(jin)
                netconfig.update(contents["network"])
                EPOCHS = contents["num_epochs"]

        training_stats = {
            "mean_abs_error": [-1] * args.num_runs,
            "max_abs_error": [-1] * args.num_runs,
            "mean_abs_error_on_surface": [-1] * args.num_runs,
            "max_abs_error_on_surface": [-1] * args.num_runs,
            "mean_abs_error_off_surface": [-1] * args.num_runs,
            "max_abs_error_off_surface": [-1] * args.num_runs,
            "mean_normal_alignment": [-1] * args.num_runs,
            "max_normal_alignment": [-1] * args.num_runs,
            "mean_curvature_error": [-1] * args.num_runs,
            "max_curvature_error": [-1] * args.num_runs,
            "execution_times": [-1] * args.num_runs
        }
        i = 0
        while i < args.num_runs:
            model = SIREN(3, 1, **netconfig)
            print(model)
            print("# parameters =", parameters_to_vector(model.parameters()).numel())
            optim = torch.optim.Adam(lr=1e-3, params=model.parameters())

            # Training the model
            model.train()
            start = time.time()
            training_loss = {}
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
                training_normals, _ = grad_sdf(training_pts, mesh_model, no_curv=True)
                training_sdf = torch.cat((
                    torch.zeros(surf_pts.shape[0]),
                    -1 * torch.ones(domain_pts.shape[0])
                ))

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

                # if not e % 10 and e > 0:
                #     print(f"Epoch {e} --- Loss: {running_loss.item()}")

                running_loss.backward()
                optim.step()

            tot_time = time.time() - start

            # fig, ax = plt.subplots(1)
            # for k in training_loss:
            #     ax.plot(list(range(EPOCHS)), training_loss[k], label=k)

            # fig.legend()
            # plt.savefig(f"loss_siren_{args.input}_{i}.png")
            # plt.show()

            # Testing the model
            # Inference on the test data.
            if isinstance(test_pts, np.ndarray):
                test_pts = torch.from_numpy(test_pts).float()

            if isinstance(test_pts, np.ndarray):
                test_sdf = torch.from_numpy(test_sdf).float()

            model.eval()
            n_siren, curv_siren = grad_sdf(test_surf_pts, model)
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

                errs_curv = torch.abs(test_curvatures - curv_siren)

                print(f"SIREN Results:"
                      f" MABSE {errs.mean():.3} -- MAXERR {errs.max().item():.3}"
                      f" --- Runtime {tot_time:} s")
                print(f"Errors on surface --- "
                      f"MABSE {errs_on_surf.mean():.3} -- MAXERR {errs_on_surf.max().item():.3}")
                print(f"Errors off surface --- "
                      f"MABSE {errs_off_surf.mean():.3} -- MAXERR {errs_off_surf.max().item():.3}")
                print(f"Normal alignment errors --- MEAN {errs_normals.mean():.3}"
                      f" --- MAX {errs_normals.max().item():.3}")

                training_stats["execution_times"][i] = tot_time
                training_stats["max_abs_error"][i] = errs.max().item()
                training_stats["mean_abs_error"][i] = errs.mean().item()
                training_stats["max_abs_error_on_surface"][i] = errs_on_surf.max().item()
                training_stats["mean_abs_error_on_surface"][i] = errs_on_surf.mean().item()
                training_stats["max_abs_error_off_surface"][i] = errs_off_surf.max().item()
                training_stats["mean_abs_error_off_surface"][i] = errs_off_surf.mean().item()
                training_stats["max_normal_alignment"][i] = errs_normals.max().item()
                training_stats["mean_normal_alignment"][i] = errs_normals.mean().item()
                training_stats["max_curvature_error"][i] = errs_curv.max().item()
                training_stats["mean_curvature_error"][i] = errs_curv.mean().item()
                i += 1

        # Marching cubes
        samples_sdf = model(samples[:, :3])["model_out"].reshape([args.mc_resolution] * 3)
        verts, faces, _, _ = convert_sdf_samples_to_ply(
            samples_sdf.detach(), [-1, -1, -1], voxel_size, None, None
        )
        save_ply(verts, faces, f"mc_siren_{args.input}.ply")
        pd.DataFrame.from_dict(training_stats).to_csv(
            f"stats_siren_{args.input}.csv",
            sep=";",
            index=False
        )
        if args.save_weights:
            torch.save(model.state_dict(), f"weights_siren_{args.input}.pth")

    if "i3d" in args.methods:
        netconfig = netconfig_map.get(args.input, netconfig_map["default"])
        if args.i3d_config is not None and \
           osp.exists(path := osp.expanduser(args.i3d_config)):
            with open(path, "r") as jin:
                contents = json.load(jin)
                netconfig.update(contents["network"])
                EPOCHS = contents["num_epochs"]

        training_stats = {
            "mean_abs_error": [-1] * args.num_runs,
            "max_abs_error": [-1] * args.num_runs,
            "mean_abs_error_on_surface": [-1] * args.num_runs,
            "max_abs_error_on_surface": [-1] * args.num_runs,
            "mean_abs_error_off_surface": [-1] * args.num_runs,
            "max_abs_error_off_surface": [-1] * args.num_runs,
            "mean_normal_alignment": [-1] * args.num_runs,
            "max_normal_alignment": [-1] * args.num_runs,
            "mean_curvature_error": [-1] * args.num_runs,
            "max_curvature_error": [-1] * args.num_runs,
            "execution_times": [-1] * args.num_runs
        }
        i = 0
        while i < args.num_runs:
            model = SIREN(3, 1, **netconfig)
            print(model)
            print("# parameters =", parameters_to_vector(model.parameters()).numel())
            optim = torch.optim.Adam(lr=1e-3, params=model.parameters())

            # Training the model
            model.train()
            start = time.time()
            training_loss = {}
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
                training_normals, _ = grad_sdf(training_pts, mesh_model, no_curv=True)
                with torch.no_grad():
                    training_sdf = mesh_model(training_pts)["model_out"]

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

                # if not e % 10 and e > 0:
                #     print(f"Epoch {e} --- Loss: {running_loss.item()}")

                running_loss.backward()
                optim.step()

            tot_time = time.time() - start

            # fig, ax = plt.subplots(1)
            # for k in training_loss:
            #     ax.plot(list(range(EPOCHS)), training_loss[k], label=k)

            # fig.legend()
            # plt.savefig(f"loss_i3d_{args.input}_{i}.png")
            # plt.show()

            # Testing the model
            # Inference on the test data.
            if isinstance(test_pts, np.ndarray):
                test_pts = torch.from_numpy(test_pts).float()

            if isinstance(test_pts, np.ndarray):
                test_sdf = torch.from_numpy(test_sdf).float()

            model.eval()
            n_i3d, curv_i3d = grad_sdf(test_surf_pts, model)
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

                errs_curv = torch.abs(test_curvatures - curv_i3d)

                print(f"i3d Results:"
                      f" MABSE {errs.mean():.3} -- MAXERR {errs.max().item():.3}"
                      f" --- Runtime {tot_time:} s")
                print(f"Errors on surface --- "
                      f"MABSE {errs_on_surf.mean():.3} -- MAXERR {errs_on_surf.max().item():.3}")
                print(f"Errors off surface --- "
                      f"MABSE {errs_off_surf.mean():.3} -- MAXERR {errs_off_surf.max().item():.3}")
                print(f"Normal alignment errors --- MEAN {errs_normals.mean():.3}"
                      f" --- MAX {errs_normals.max().item():.3}")

                training_stats["execution_times"][i] = tot_time
                training_stats["max_abs_error"][i] = errs.max().item()
                training_stats["mean_abs_error"][i] = errs.mean().item()
                training_stats["max_abs_error_on_surface"][i] = errs_on_surf.max().item()
                training_stats["mean_abs_error_on_surface"][i] = errs_on_surf.mean().item()
                training_stats["max_abs_error_off_surface"][i] = errs_off_surf.max().item()
                training_stats["mean_abs_error_off_surface"][i] = errs_off_surf.mean().item()
                training_stats["max_normal_alignment"][i] = errs_normals.max().item()
                training_stats["mean_normal_alignment"][i] = errs_normals.mean().item()
                training_stats["max_curvature_error"][i] = errs_curv.max().item()
                training_stats["mean_curvature_error"][i] = errs_curv.mean().item()
                i += 1

        # Marching cubes
        samples_sdf = model(samples[:, :3])["model_out"].reshape([args.mc_resolution] * 3)
        verts, faces, _, _ = convert_sdf_samples_to_ply(
            samples_sdf.detach(), [-1, -1, -1], voxel_size, None, None
        )
        save_ply(verts, faces, f"mc_i3d_{args.input}.ply")
        pd.DataFrame.from_dict(training_stats).to_csv(
            f"stats_i3d_{args.input}.csv",
            sep=";",
            index=False
        )
        if args.save_weights:
            torch.save(model.state_dict(), f"weights_i3d_{args.input}.pth")

    if "i3dcurv" in args.methods:
        netconfig = netconfig_map.get(args.input, netconfig_map["default"])
        if args.i3d_config is not None and \
           osp.exists(path := osp.expanduser(args.i3d_config)):
            with open(path, "r") as jin:
                contents = json.load(jin)
                netconfig.update(contents["network"])
                EPOCHS = contents["num_epochs"]

        training_stats = {
            "mean_abs_error": [-1] * args.num_runs,
            "max_abs_error": [-1] * args.num_runs,
            "mean_abs_error_on_surface": [-1] * args.num_runs,
            "max_abs_error_on_surface": [-1] * args.num_runs,
            "mean_abs_error_off_surface": [-1] * args.num_runs,
            "max_abs_error_off_surface": [-1] * args.num_runs,
            "mean_normal_alignment": [-1] * args.num_runs,
            "max_normal_alignment": [-1] * args.num_runs,
            "mean_curvature_error": [-1] * args.num_runs,
            "max_curvature_error": [-1] * args.num_runs,
            "execution_times": [-1] * args.num_runs
        }
        i = 0
        while i < args.num_runs:
            model = SIREN(3, 1, **netconfig)
            print(model)
            print("# parameters =", parameters_to_vector(model.parameters()).numel())
            optim = torch.optim.Adam(lr=1e-3, params=model.parameters())

            # Training the model
            model.train()
            start = time.time()
            training_loss = {}
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
                training_normals, tranining_curvatures = grad_sdf(training_pts, mesh_model)
                with torch.no_grad():
                    training_sdf = mesh_model(training_pts)["model_out"]

                gt = {
                    "sdf": training_sdf.float().unsqueeze(1),
                    "normals": training_normals.float(),
                    "curvature": tranining_curvatures.float(),
                }

                optim.zero_grad()

                y = model(training_pts)
                loss = true_sdf_curvature(y, gt)

                running_loss = torch.zeros((1, 1))
                for k, v in loss.items():
                    running_loss += v
                    if k not in training_loss:
                        training_loss[k] = [v.item()]
                    else:
                        training_loss[k].append(v.item())

                # if not e % 10 and e > 0:
                #     print(f"Epoch {e} --- Loss: {running_loss.item()}")

                running_loss.backward()
                optim.step()

            tot_time = time.time() - start

            # fig, ax = plt.subplots(1)
            # for k in training_loss:
            #     ax.plot(list(range(EPOCHS)), training_loss[k], label=k)

            # fig.legend()
            # plt.savefig(f"loss_i3dcurv_{args.input}_{i}.png")

            # Testing the model
            # Inference on the test data.
            if isinstance(test_pts, np.ndarray):
                test_pts = torch.from_numpy(test_pts).float()

            if isinstance(test_pts, np.ndarray):
                test_sdf = torch.from_numpy(test_sdf).float()

            model.eval()
            n_i3d, curv_i3d = grad_sdf(test_surf_pts, model)
            with torch.no_grad():
                y_i3d = model(test_pts)["model_out"].squeeze()
                errs = torch.abs(test_sdf - y_i3d)
                errs_on_surf = errs[:test_surf_pts.shape[0]]
                errs_off_surf = errs[test_surf_pts.shape[0]:]

                # 1 - F.cosine_similarity(pred_vectors, gt_vectors, dim=-1)[..., None]
                errs_normals = 1 - F.cosine_similarity(
                    test_normals[:test_surf_pts.shape[0], ...],
                    n_i3d,
                    dim=-1
                )

                errs_curv = torch.abs(test_curvatures - curv_i3d)

                print(f"i3dcurv Results:"
                      f" MABSE {errs.mean():.3} -- MAXERR {errs.max().item():.3}"
                      f" --- Runtime {tot_time:} s")
                print(f"Errors on surface --- "
                      f"MABSE {errs_on_surf.mean():.3} -- MAXERR {errs_on_surf.max().item():.3}")
                print(f"Errors off surface --- "
                      f"MABSE {errs_off_surf.mean():.3} -- MAXERR {errs_off_surf.max().item():.3}")
                print(f"Normal alignment errors --- MEAN {errs_normals.mean():.3}"
                      f" --- MAX {errs_normals.max().item():.3}")

                training_stats["execution_times"][i] = tot_time
                training_stats["max_abs_error"][i] = errs.max().item()
                training_stats["mean_abs_error"][i] = errs.mean().item()
                training_stats["max_abs_error_on_surface"][i] = errs_on_surf.max().item()
                training_stats["mean_abs_error_on_surface"][i] = errs_on_surf.mean().item()
                training_stats["max_abs_error_off_surface"][i] = errs_off_surf.max().item()
                training_stats["mean_abs_error_off_surface"][i] = errs_off_surf.mean().item()
                training_stats["max_normal_alignment"][i] = errs_normals.max().item()
                training_stats["mean_normal_alignment"][i] = errs_normals.mean().item()
                training_stats["max_curvature_error"][i] = errs_curv.max().item()
                training_stats["mean_curvature_error"][i] = errs_curv.mean().item()
                i += 1

        # Marching cubes
        samples_sdf = model(samples[:, :3])["model_out"].reshape([args.mc_resolution] * 3)
        verts, faces, _, _ = convert_sdf_samples_to_ply(
            samples_sdf.detach(), [-1, -1, -1], voxel_size, None, None
        )
        save_ply(verts, faces, f"mc_i3dcurv_{args.input}.ply")
        pd.DataFrame.from_dict(training_stats).to_csv(
            f"stats_i3dcurv_{args.input}.csv",
            sep=";",
            index=False
        )
        if args.save_weights:
            torch.save(model.state_dict(), f"weights_i3dcurv_{args.input}.pth")
