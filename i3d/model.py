# coding: utf-8

from collections import OrderedDict
import os.path as osp
import torch
from torch import nn
import numpy as np


@torch.no_grad()
def sine_init(m, w0):
    if hasattr(m, 'weight'):
        num_input = m.weight.size(-1)
        m.weight.uniform_(-np.sqrt(6 / num_input) / w0,
                          np.sqrt(6 / num_input) / w0)


@torch.no_grad()
def first_layer_sine_init(m):
    if hasattr(m, 'weight'):
        num_input = m.weight.size(-1)
        m.weight.uniform_(-1 / num_input, 1 / num_input)


class SineLayer(nn.Module):
    """A Sine non-linearity layer.
    """
    def __init__(self, w0=30):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)

    def __repr__(self):
        return f"SineLayer(w0={self.w0})"


class SIREN(nn.Module):
    """SIREN Module

    Parameters
    ----------
    n_in_features: int
        Number of input features.

    n_out_features: int
        Number of output features.

    hidden_layer_config: list[int], optional
        Number of neurons at each hidden layer of the network. The model will
        have `len(hidden_layer_config)` hidden layers. Only used in during
        model training. Default value is None.

    w0: number, optional
        Frequency multiplier for the input Sine layers. Only useful for
        training the model. Default value is 30, as per [1].

    ww: number, optional
        Frequency multiplier for the hidden Sine layers. Only useful for
        training the model. Default value is None.

    delay_init: boolean, optional
        Indicates if we should perform the weight initialization or not.
        Default value is False, meaning that we perform the weight
        initialization as usual. This is useful if we will load the weights of
        a pre-trained network, in this case, initializing the weights does not
        make sense, since they will be overwritten.

    References
    ----------
    [1] Sitzmann, V., Martel, J. N. P., Bergman, A. W., Lindell, D. B.,
    & Wetzstein, G. (2020). Implicit Neural Representations with Periodic
    Activation Functions. ArXiv. http://arxiv.org/abs/2006.09661
    """
    def __init__(self, n_in_features, n_out_features, hidden_layer_config=[],
                 w0=30, ww=None, delay_init=False):
        super(SIREN, self).__init__()
        self.in_features = n_in_features
        self.out_features = n_out_features
        self.w0 = w0
        if ww is None:
            self.ww = w0
        else:
            self.ww = ww

        net = []
        net.append(nn.Sequential(
            nn.Linear(n_in_features, hidden_layer_config[0]),
            SineLayer(self.w0)
        ))

        for i in range(1, len(hidden_layer_config)):
            net.append(nn.Sequential(
                nn.Linear(hidden_layer_config[i-1], hidden_layer_config[i]),
                SineLayer(self.ww)
            ))

        net.append(nn.Sequential(
            nn.Linear(hidden_layer_config[-1], n_out_features),
        ))

        self.net = nn.Sequential(*net)
        if not delay_init:
            self.reset_weights()

    def forward(self, x, preserve_grad=False):
        """Forward pass of the model.

        Parameters
        ----------
        x: torch.Tensor
            The model input containing of size Nx3

        Returns
        -------
        dict
            Dictionary of tensors with the input coordinates under 'model_in'
            and the model output under 'model_out'.
        """
        # Enables us to compute gradients w.r.t. coordinates
        if preserve_grad:
            coords_org = x
        else:
            coords_org = x.clone().detach().requires_grad_(True)

        coords = coords_org
        y = self.net(coords)
        return {"model_in": coords_org, "model_out": y}

    def reset_weights(self):
        """Resets the weights of the network using Sitzmann et al. (2020).

        Returns
        -------
        self: nifm.model.SIREN
            The network.
        """
        self.net[0].apply(first_layer_sine_init)
        self.net[1:].apply(lambda module: sine_init(module, self.ww))
        return self

    def update_omegas(self, w0=1, ww=None):
        """Updates the omega values for all layers except the last one.

        Note that this updates the w0 and ww instance attributes.

        Parameters
        ----------
        w0: number, optional
            The new omega_0 value to assume. By default is 1.

        ww: number, optional
            The new omega_w value to assume. By default is None, meaning
            that `ww` wil be set to `w0`.
        """
        if ww is None:
            ww = w0

        # Updating the state_dict weights and biases.
        my_sd = self.state_dict()
        keys = list(my_sd.keys())
        for k in keys[:-2]:
            my_sd[k] = my_sd[k] * (self.w0 / w0)

        self.load_state_dict(my_sd)
        self.w0 = w0
        self.ww = ww

        # Updating the omega values of the SineLayer instances
        self.net[0][1].w0 = w0
        for i in range(1, len(self.net)-1):
            self.net[i][1].w0 = ww

        return self


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


def from_state_dict(weights: OrderedDict, device: str = "cpu", w0=1, ww=None):
    """Builds a SIREN network with the topology and weights in `weights`.

    Parameters
    ----------
    weights: OrderedDict
        The input state_dict to use as reference.

    device: str, optional
        Device to load the weights. Default value is cpu.

    w0: number, optional
        Frequency parameter for the first layer. Default value is 1.

    ww: number, optional
        Frequency parameter for the intermediate layers. Default value is None,
        we will assume that ww = w0 in this case

    Returns
    -------
    model: nifm.model.SIREN
        The NN model mirroring `weights` topology.

    upgrade_to_v2: boolean
        If `weights` was from an older version of SIREN, we convert them to our
        format and set this to `True`, signaling this fact.
    """
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
    upgrade_to_v2 = False
    try:
        model.load_state_dict(weights)
    except RuntimeError:
        print("Found weights from old version of SIREN. Converting to v2.")
        new_weights, diff = siren_v1_to_v2(weights, True)
        model.load_state_dict(new_weights)
        upgrade_to_v2 = True

    return model, upgrade_to_v2


def from_pth(path, device="cpu", w0=1, ww=None):
    """Builds a SIREN given the path to a weights file.

    Note that if the weights pointed by `path` correspond to an older version
    of SIREN, we convert it to our format and save this converted weights
    file as `path.split(".")[0]+"_v2.pth"`.

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
    model, upgraded_to_v2 = from_state_dict(
        weights, device=device, w0=w0, ww=ww
    )
    if upgraded_to_v2:
        torch.save(model.state_dict(), path.split(".")[0] + "_v2.pth")

    return model.to(device=device)
