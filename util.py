# coding: utf-8

from collections import OrderedDict
import json
import os
import os.path as osp
import shutil
import logging
import torch
from model import SIREN


def create_output_paths(checkpoint_path, experiment_name, overwrite=True):
    """Helper function to create the output folders. Returns the resulting path.
    """
    full_path = osp.join(".", checkpoint_path, experiment_name)
    if osp.exists(full_path) and overwrite:
        shutil.rmtree(full_path)
    elif osp.exists(full_path):
        logging.warning("Output path exists. Not overwritting.")
        return full_path

    os.makedirs(osp.join(full_path, "models"))
    os.makedirs(osp.join(full_path, "reconstructions"))
    return full_path


def load_experiment_parameters(parameters_path):
    try:
        with open(parameters_path, "r") as fin:
            parameter_dict = json.load(fin)
    except FileNotFoundError:
        logging.warning("File '{parameters_path}' not found.")
        return {}
    return parameter_dict


def siren_v1_to_v2(model_in, check_equals=False):
    """Converts the models trained using the old class to the new format.

    Parameters
    ----------
    model_in: OrderedDict
        Model trained by our old SIREN version (Sitzmann code).

    check_equals: boolean, optional
        Whether to check if the converted models weight match. By default this
        is False.

    Returns
    -------
    model_out: OrderedDict
        The input model converted to a format recognizable by our version of
        SIREN.

    divergences: list[tuple[str, str]]
        If `check_equals` is True, then this list contains the keys where the
        original and converted model dictionaries are not equal. Else, this is
        an empty list.

    See Also
    --------
    `model.SIREN`
    """
    model_out = OrderedDict()
    for k, v in model_in.items():
        model_out[k[4:]] = v

    divergences = []
    if check_equals:
        for k in model_in.keys():
            test = model_in[k] == model_out[k[4:]]
            if test.sum().item() != test.numel():
                divergences.append((k, k[4:]))

    return model_out, divergences


def from_pth(path, device="cpu", w0=1, ww=None):
    """Builds a SIREN given a weights file.

    Parameters
    ----------
    path: str
        Path to the pth file.

    device: str, optional
        Device to load the weights. Default value is cpu.

    w0: number, optional
        Frequency parameter for the first layer. Default value is 1.

    ww: number, optional
        Frequency parameter for the intermediate layers. Default value is None,
        we will assume that ww = w0 in this case.

    Returns
    -------
    model: torch.nn.Module
        The resulting model.

    Raises
    ------
    FileNotFoundError if `path` points to a non-existing file.
    """
    if not osp.exists(path):
        raise FileNotFoundError(f"Weights file not found at \"{path}\"")

    weights = torch.load(path, map_location=torch.device(device))
    # Each layer has two tensors, one for weights other for biases.
    n_layers = len(weights) // 2
    hidden_layer_config = [None] * (n_layers - 1)
    keys = list(weights.keys())

    bias_keys = [k for k in keys if "bias" in k]
    i = 0
    while i < (n_layers - 1):
        k = bias_keys[i]
        hidden_layer_config[i] = weights[k].shape[0]
        i += 1

    n_in_features = weights[keys[0]].shape[1]
    n_out_features = weights[keys[-1]].shape[0]
    model = SIREN(
        n_in_features=n_in_features,
        n_out_features=n_out_features,
        hidden_layer_config=hidden_layer_config,
        w0=w0, ww=ww, delay_init=True
    )

    # Loads the weights. Converts to version 2 if they are from the old version
    # of SIREN.
    try:
        model.load_state_dict(weights)
    except RuntimeError:
        print("Found weights from old version of SIREN. Converting to v2.")
        new_weights, diff = siren_v1_to_v2(weights, True)
        new_weights_file = path.split(".")[0] + "_v2.pth"
        torch.save(new_weights, new_weights_file)
        model.load_state_dict(new_weights)

    return model
