# coding: utf-8

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
        super().__init__()
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
            self.net[0].apply(first_layer_sine_init)
            self.net[1:].apply(lambda module: sine_init(module, self.ww))

    def forward(self, x):
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
        coords_org = x.clone().detach().requires_grad_(True)
        coords = coords_org
        y = self.net(coords)
        return {"model_in": coords_org, "model_out": y}
