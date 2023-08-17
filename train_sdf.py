#!/usr/bin/env python
# coding: utf-8

"""
Experiments with calculating the SDF for a batch of points and reusing it for N
iterations.
"""

import argparse
import copy
import os
import os.path as osp
import time
import yaml
import numpy as np
import torch
from torch.nn.utils import parameters_to_vector
from i3d.dataset import PointCloudDeferredSampling
from i3d.loss_functions import true_sdf
from i3d.model import SIREN


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
    parser.add_argument(
        "--hidden-layer-config", type=int, nargs='+', default=[],
        help="SIREN neurons per layer. By default we fetch it from the"
        " configuration file."
    )
    parser.add_argument(
        "--batchsize", "-b", type=int, default=0,
        help="# of points to fetch per iteration. By default, uses the # of"
        " mesh vertices."
    )
    parser.add_argument(
        "--resample-sdf-at", "-r", type=int, default=0,
        help="Recalculates the SDF for off-surface points at every N epochs."
        " By default (0) we calculate the SDF at every iteration."
    )
    parser.add_argument(
        "--sampling", "-s", type=str, default="uniform",
        help="Uniform (\"uniform\", default value) or curvature-based"
        " (\"curvature\") sampling."
    )
    parser.add_argument(
        "--curvature-fractions", type=float, nargs='+', default=[],
        help="Fractions of data to fetch for each curvature bin. Only used"
        " with \"--sampling curvature\" argument, or when sampling type is"
        " \"curvature\" in the configuration file."
    )
    parser.add_argument(
        "--curvature-percentiles", type=float, nargs='+', default=[],
        help="The curvature percentiles to use when defining the bins. Only"
        " used with \"--sampling curvature\" argument, or when sampling type"
        " is \"curvature\" in the configuration file."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="RNG seed to use."
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

    seed = args.seed if args.seed else config["training"].get("seed", 668123)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    config["training"]["seed"] = seed

    trainingcfg = config["training"]
    EPOCHS = trainingcfg.get("epochs", 100)
    if args.nepochs:
        EPOCHS = args.nepochs
        config["training"]["epochs"] = args.nepochs

    BATCH = trainingcfg.get("batchsize", 0)
    if args.batchsize:
        BATCH = args.batchsize
        config["training"]["batchsize"] = args.batchsize

    REFRESH_SDF_AT_PERC_STEPS = trainingcfg.get("resample_sdf_at", 1)
    if args.resample_sdf_at:
        REFRESH_SDF_AT_PERC_STEPS = args.resample_sdf_at
        config["training"]["resample_sdf_at"] = args.resample_sdf_at

    REFRESH_SDF_AT_PERC_STEPS /= EPOCHS

    devstr = args.device
    if "cuda" in devstr and not torch.cuda.is_available():
        devstr = "cpu"
        print("No CUDA available devices found on system. Using CPU.")

    device = torch.device(devstr)

    withcurvature = False
    if "sampling" not in config:
        config["sampling"] = {"type": "uniform"}
    elif config["sampling"]["type"] == "curvature":
        withcurvature = True

    if args.sampling == "curvature":
        withcurvature = True
        config["sampling"]["type"] = "curvature"

    curvature_fractions = []
    curvature_percentiles = []
    if withcurvature:
        curvature_fractions = config["sampling"].get(
            "curvature_fractions", [0.2, 0.6, 0.2]
        )
        curvature_percentiles = config["sampling"].get(
            "curvature_percentiles", [0.7, 0.95]
        )
        if args.curvature_fractions:
            curvature_fractions = [float(f) for f in args.curvature_fractions]
            config["sampling"]["curvature_fractions"] = curvature_fractions
        if args.curvature_percentiles:
            curvature_percentiles = \
                [float(p) for p in args.curvature_percentiles]
            config["sampling"]["curvature_percentiles"] = curvature_percentiles

    dataset = PointCloudDeferredSampling(
        args.meshpath, batch_size=BATCH, use_curvature=withcurvature,
        device=device, curvature_fractions=curvature_fractions,
        curvature_percentiles=curvature_percentiles
    )
    N = dataset.vertices.shape[0]

    # Fetching batch_size again since we may have passed 0, meaning that we
    # will use all mesh vertices at each iteration.
    BATCH = dataset.batch_size
    nsteps = round(EPOCHS * (2 * N / BATCH))
    warmup_steps = nsteps // 10
    resample_sdf_nsteps = max(1, round(REFRESH_SDF_AT_PERC_STEPS * nsteps))
    print(f"Resampling SDF at every {resample_sdf_nsteps} training steps")
    print(f"Total # of training steps = {nsteps}")

    netcfg = config["network"]
    hidden_layer_config = netcfg["hidden_layers"]
    if args.hidden_layer_config:
        hidden_layer_config = [int(n) for n in args.hidden_layer_config]
        config["network"]["hidden_layers"] = hidden_layer_config

    # Create the model and optimizer
    model = SIREN(
        netcfg["in_coords"],
        netcfg["out_coords"],
        hidden_layer_config=hidden_layer_config,
        w0=netcfg["omega_0"] if not args.omega0 else args.omega0,
        ww=netcfg["omega_w"] if not args.omegaW else args.omegaW
    ).to(device)
    print(model)
    print("# parameters =", parameters_to_vector(model.parameters()).numel())
    optim = torch.optim.Adam(lr=1e-4, params=model.parameters())
    training_loss = {}

    best_loss = torch.inf
    best_weights = None
    best_step = warmup_steps

    config["network"]["omega_0"] = model.w0
    config["network"]["omega_w"] = model.ww

    with open(osp.join(args.outputpath, "config.yaml"), 'w') as fout:
        yaml.dump(config, fout)

    # Training loop
    start_training_time = time.time()
    for step in range(nsteps):
        # We will recalculate the SDF points at this # of steps
        if not step % resample_sdf_nsteps:
            dataset.refresh_sdf = True

        samples = dataset[0]
        gt = samples[1]

        optim.zero_grad(set_to_none=True)
        y = model(samples[0]["coords"])
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
            best_step = step
            best_weights = copy.deepcopy(model.state_dict())
            best_loss = running_loss.item()

        if not step % 100 and step > 0:
            print(f"Step {step} --- Loss {running_loss.item()}")

    training_time = time.time() - start_training_time
    print(f"Training took {training_time} s")
    print(f"Best loss value {best_loss} at step {best_step}")
    torch.save(
        model.state_dict(), osp.join(args.outputpath, "weights_with_w0.pth")
    )
    model.update_omegas(w0=1, ww=None)
    torch.save(
        model.state_dict(), osp.join(args.outputpath, "weights.pth")
    )
    torch.save(
        best_weights, osp.join(args.outputpath, "best_with_w0.pth")
    )

    model.w0 = netcfg["omega_0"] if not args.omega0 else args.omega0
    model.ww = netcfg["omega_w"] if not args.omegaW else args.omegaW
    model.load_state_dict(best_weights)
    model.update_omegas(w0=1, ww=None)
    torch.save(
        model.state_dict(), osp.join(args.outputpath, "best.pth")
    )
