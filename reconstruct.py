#!/usr/bin/env python
# coding: utf-8

"""
Simple script to run multiple SDF reconstructions given a base log directory
and a set of checkpoints.
"""

import argparse
import json
import logging
import os
import torch
from meshing import create_mesh
from model import SIREN


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run marching cubes using a trained model.")
    parser.add_argument(
        "experiment_path",
        help="Path to the JSON experiment description file"
    )
    parser.add_argument(
        "--checkpoints", "-c", nargs="+", default=["final"],
        help="Checkpoints to use when reconstructing the model."
    )

    args = parser.parse_args()
    with open(args.experiment_path, "r") as fin:
        params = json.load(fin)

    base_path = os.path.join(params["checkpoint_path"], params["experiment_name"])
    model_path = os.path.join(base_path, "models")
    dest_path = os.path.join(base_path, "reconstructions")

    model = SIREN(
        n_in_features=3,
        n_out_features=1,
        hidden_layer_config=params["network"]["hidden_layer_nodes"],
        w0=params["network"]["w0"]
    )

    for c in args.checkpoints:
        logging.info(f"Marching cubes running for checkpoint \"{c}\"")

        checkpoint_file = os.path.join(model_path, f"model_{c}.pth")
        if not os.path.exists(checkpoint_file):
            logging.warning(f"Checkpoint file model_{c}.pth does not exist. Skipping")
            continue

        model.load_state_dict(
            torch.load(checkpoint_file)
        )
        create_mesh(
            model,
            os.path.join(dest_path, f"{c}.ply"),
            N=params["reconstruction"]["resolution"]
        )

    logging.info("Done")
