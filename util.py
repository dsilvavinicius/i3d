# coding: utf-8

import json
import os
import shutil
import logging


def create_output_paths(checkpoint_path, experiment_name, overwrite=True):
    """Helper function to create the output folders. Returns the resulting path.
    """
    full_path = os.path.join(".", checkpoint_path, experiment_name)
    if os.path.exists(full_path) and overwrite:
        shutil.rmtree(full_path)
    elif os.path.exists(full_path):
        logging.warning("Output path exists. Not overwritting.")
        return full_path

    os.makedirs(os.path.join(full_path, "models"))
    os.makedirs(os.path.join(full_path, "reconstructions"))
    return full_path


def load_experiment_parameters(parameters_path):
    try:
        with open(parameters_path, "r") as fin:
            parameter_dict = json.load(fin)
    except FileNotFoundError:
        logging.warning("File '{parameters_path}' not found.")
        return {}
    return parameter_dict
