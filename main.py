#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import logging
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import PointCloud
from loss_functions import sdf_sitzmann, true_sdf_curvature, true_sdf
from meshing import create_mesh
from model import SIREN
from util import create_output_paths, load_experiment_parameters


def train_model(dataset, model, device, config, silent=False):
    BATCH_SIZE = config["batch_size"]
    EPOCHS = config["epochs"]

    EPOCHS_TIL_CHECKPOINT = config.get("epochs_to_checkpoint", 0)
    EPOCHS_TIL_RECONSTRUCTION = config.get("epochs_to_reconstruct", 0)

    log_path = config["log_path"]
    loss_fn = config["loss_fn"]
    optim = config["optimizer"]
    sampler = config.get("sampler", None)
    if sampler is not None:
        train_loader = DataLoader(
            dataset,
            batch_sampler=BatchSampler(sampler, batch_size=BATCH_SIZE, drop_last=False),
            pin_memory=True,
            num_workers=0
        )
    else:
        train_loader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=1,
            pin_memory=True,
            num_workers=0
        )
    model.to(device)

    # Creating the summary storage folder
    summary_path = os.path.join(log_path, 'summaries')
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    writer = SummaryWriter(summary_path)

    losses = dict()
    for epoch in range(EPOCHS):
        running_loss = dict()
        for i, (input_data, gt_data) in enumerate(train_loader, start=0):
            # If we have a custom sampler, we must reshape the Tensors from
            # [B, N, D] to [1, B*N, D]
            # if sampler is not None:
            #     for k, v in data.items():
            #         b, n, d = v.size()
            #         data[k] = v.reshape(1, -1, d)

            # get the inputs; data is a list of [inputs, labels]
            inputs = {k: v.to(device) for k, v in input_data.items()}
            gt = {k: v.to(device) for k, v in gt_data.items()}

            # zero the parameter gradients
            optim.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs["coords"])
            loss = loss_fn(outputs, gt)

            train_loss = torch.zeros((1, 1), device=device)
            for it, l in loss.items():
                train_loss += l
                # accumulating statistics per loss term
                if it not in running_loss:
                    running_loss[it] = l.item()
                else:
                    running_loss[it] += l.item()

            train_loss.backward()
            optim.step()

        # accumulate statistics
        for it, l in running_loss.items():
            if it in losses:
                losses[it][epoch] = l
            else:
                losses[it] = [0.] * EPOCHS
                losses[it][epoch] = l
            writer.add_scalar(it, l, epoch)

        if not silent:
            epoch_loss = 0
            for k, v in running_loss.items():
                epoch_loss += v
            print(f"Epoch: {epoch} - Loss: {epoch_loss}")

        # saving the model at checkpoints
        if epoch and EPOCHS_TIL_CHECKPOINT and not \
           epoch % EPOCHS_TIL_CHECKPOINT:
            if not silent:
                print(f"Saving model for epoch {epoch}")
            torch.save(
                model.state_dict(),
                os.path.join(log_path, "models", f"model_{epoch}.pth")
            )
        else:
            torch.save(
                model.state_dict(),
                os.path.join(log_path, "models", "model_current.pth")
            )

        if epoch and EPOCHS_TIL_RECONSTRUCTION and not \
           epoch % EPOCHS_TIL_RECONSTRUCTION:
            if not silent:
                print(f"Reconstructing mesh for epoch {epoch}")
            create_mesh(
                model,
                os.path.join(log_path, "reconstructions", f"{epoch}.ply"),
                N=config["mc_resolution"],
                device=device
            )
            model.train()

    return losses


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        usage="python main.py path_to_experiments.json"
    )

    p.add_argument(
        "experiment_path",
        help="Path to the JSON experiment description file"
    )
    p.add_argument(
        "-s", "--silent", action="store_true",
        help="Suppresses informational output messages"
    )
    args = p.parse_args()
    parameter_dict = load_experiment_parameters(args.experiment_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sampling_config = parameter_dict["sampling_opts"]

    full_path = create_output_paths(
        parameter_dict["checkpoint_path"],
        parameter_dict["experiment_name"],
        overwrite=False
    )

    n_in_features = 3  # implicit 3D models

    # Saving the parameters to the output path
    with open(os.path.join(full_path, "params.json"), "w+") as fout:
        json.dump(parameter_dict, fout, indent=4)

    no_sampler = True
    if sampling_config.get("sampler"):
        no_sampler = False

    off_surface_sdf = parameter_dict.get("off_surface_sdf", None)
    off_surface_normals = parameter_dict.get("off_surface_normals", None)
    scaling = parameter_dict.get("scaling", None)
    dataset = PointCloud(
        os.path.join("data", parameter_dict["dataset"]),
        samples_on_surface=sampling_config["samples_on_surface"],
        scaling=scaling,
        off_surface_sdf=off_surface_sdf,
        off_surface_normals=off_surface_normals,
        random_surf_samples=sampling_config["random_surf_samples"],
        no_sampler=no_sampler,
        batch_size=parameter_dict["batch_size"],
        uniform_sampling=sampling_config.get("uniform_sampling", True),
        curvature_fracs=sampling_config.get("curvature_iteration_fractions", None),
        low_med_percentiles=sampling_config.get("percentile_thresholds", None)
    )

    sampler = None
    # sampler_opt = sampling_config.get("sampler", None)
    # if sampler_opt is not None and sampler_opt == "sitzmann":
    #     sampler = SitzmannSampler(
    #         dataset,
    #         sampling_config["samples_off_surface"]
    #     )

    model = SIREN(
        n_in_features,
        n_out_features=1,
        hidden_layer_config=parameter_dict["network"]["hidden_layer_nodes"],
        w0=parameter_dict["network"]["w0"]
    )
    if not args.silent:
        print(model)

    opt_params = parameter_dict["optimizer"]
    if opt_params["type"] == "adam":
        optimizer = torch.optim.Adam(
            lr=opt_params["lr"],
            params=model.parameters()
        )

    loss_opt = parameter_dict.get("loss")
    loss_fn = sdf_sitzmann
    if loss_opt == "sdf":
        loss_fn = true_sdf
    elif loss_opt == "curvature":
        loss_fn = true_sdf_curvature
    elif loss_opt == "sitzmann":
        pass  # same as default
    else:
        logging.warning("Unknown loss option. Using default \"sitzmann\".")

    config_dict = {
        "epochs": parameter_dict["num_epochs"],
        "batch_size": parameter_dict["batch_size"],
        "epochs_to_checkpoint": parameter_dict["epochs_to_checkpoint"],
        "epochs_to_reconstruct": parameter_dict["epochs_to_reconstruction"],
        "sampler": sampler,
        "log_path": full_path,
        "optimizer": optimizer,
        "loss_fn": loss_fn,
        "mc_resolution": parameter_dict["reconstruction"]["resolution"]
    }

    losses = train_model(
        dataset,
        model,
        device,
        config_dict,
        silent=args.silent
    )
    loss_df = pd.DataFrame.from_dict(losses)
    loss_df.to_csv(os.path.join(full_path, "losses.csv"), sep=";", index=None)

    # saving the final model
    torch.save(
        model.state_dict(),
        os.path.join(full_path, "models", "model_final.pth")
    )

    # reconstructing the final mesh
    mesh_file = parameter_dict["reconstruction"]["output_file"] + ".ply"
    mesh_resolution = parameter_dict["reconstruction"]["resolution"]

    create_mesh(
        model,
        os.path.join(full_path, "reconstructions", mesh_file),
        N=mesh_resolution,
        device=device
    )
