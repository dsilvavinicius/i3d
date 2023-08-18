# coding: utf-8

import itertools
import numpy as np
import torch
from torch.autograd import grad


def gaussian_curvature(grad: torch.Tensor, hess: torch.Tensor) -> torch.Tensor:
    """Calculates the Gaussian curvature of a implicit surface.
    See https://en.wikipedia.org/wiki/Gaussian_curvature#Alternative_formulas
    for details.

    Parameters
    ----------
    grad: torch.Tensor
        Gradient of the surface, shaped [B, N, 3].

    hess: torch.Tensor
        Hessian of the surface, shaped [B, N, 1, 1, 3, 3].

    Returns
    -------
    Kg: torch.Tensor
        The gaussian curvatures.
    """
    # Append gradients to the last columns of the hessians.
    grad5d = torch.unsqueeze(grad, 2)
    grad5d = torch.unsqueeze(grad5d, -1)
    F = torch.cat((hess, grad5d), -1)

    # Append gradients (with and additional 0 at the last coord) to the last
    # lines of the hessians.
    hess_size = hess.size()
    zeros_size = list(itertools.chain.from_iterable((hess_size[:3], [1, 1])))
    zeros = torch.zeros(zeros_size).to(grad.device)
    grad5d = torch.unsqueeze(grad, 2)
    grad5d = torch.unsqueeze(grad5d, -2)
    grad5d = torch.cat((grad5d, zeros), -1)

    F = torch.cat((F, grad5d), -2)
    grad_norm = torch.norm(grad, dim=-1)

    Kg = -torch.det(F)[-1].squeeze(-1) / (grad_norm[0]**4)
    return Kg


def mean_curvature(y, x):
    grad = gradient(y, x)
    grad_norm = torch.norm(grad, dim=-1)
    unit_grad = grad.squeeze(-1)/grad_norm.unsqueeze(-1)

    Km = 0.5*divergence(unit_grad, x)
    return Km


def principal_curvature(y, x, grad, hess):
    Kg = gaussian_curvature(grad, hess).unsqueeze(-1)
    Km = mean_curvature(y, x).squeeze(0)
    A = torch.sqrt(torch.abs(torch.pow(Km, 2) - Kg) + 0.00001)
    Kmax = Km + A
    Kmin = Km - A

    return Kmin, Kmax


#Che, Wujun, Jean-Claude Paul, and Xiaopeng Zhang.
#"Lines of curvature and umbilical points for implicit surfaces.
#" Computer Aided Geometric Design 24.7 (2007): 395-409.
def principal_directions(grad, hess):
    A =      grad[...,[1]]*hess[...,0,2] - grad[...,[2]]*hess[...,0,1]
    B = 0.5*(grad[...,[2]]*hess[...,0,0] - grad[...,[0]]*hess[...,0,2] + grad[...,[1]]*hess[...,1,2] - grad[...,[2]]*hess[...,1,1])
    C = 0.5*(grad[...,[1]]*hess[...,2,2] - grad[...,[2]]*hess[...,1,2] + grad[...,[0]]*hess[...,0,1] - grad[...,[1]]*hess[...,0,0])
    D =      grad[...,[2]]*hess[...,0,1] - grad[...,[0]]*hess[...,1,2]
    E = 0.5*(grad[...,[0]]*hess[...,1,1] - grad[...,[1]]*hess[...,0,1] + grad[...,[2]]*hess[...,0,2] - grad[...,[0]]*hess[...,2,2])
    F =      grad[...,[0]]*hess[...,1,2] - grad[...,[1]]*hess[...,0,2]

    U = A*grad[...,[2]]**2 - 2.*C*grad[...,[0]]*grad[...,[2]] + F*grad[...,[0]]**2
    V = 2*(B*grad[...,[2]]**2 - C*grad[...,[1]]*grad[...,[2]] - E*grad[...,[0]]*grad[...,[2]] + F*grad[...,[0]]*grad[...,[1]])
    W = D*grad[...,[2]]**2 - 2.*E*grad[...,[1]]*grad[...,[2]] + F*grad[...,[1]]**2

    # Hz signal
    s = torch.sign(grad[...,[2]])

    # U == 0 and W == 0
    UW_mask = (torch.abs(U) < 1e-10) * (torch.abs(W) < 1e-10)
    UW_mask_shape = list(UW_mask.shape)
    UW_mask_shape[-1] *= 3
    UW_mask_3 = UW_mask.expand(UW_mask_shape)

    # U != 0 or W != 0
    mask = ~UW_mask
    mask_3 = ~UW_mask_3

    # first direction (U!=0 or W!=0)
    T1x = (-V + s * torch.sqrt(torch.abs(V ** 2 - 4 * U * W) + 1e-10)) * grad[..., [2]]
    T1y = 2 * U * grad[..., [2]]
    T1z = (V - s * torch.sqrt(torch.abs(V ** 2 - 4 * U * W) + 1e-10)) * grad[..., [0]] - 2 * U * grad[..., [1]]
    dir_min = torch.cat((T1x, T1y), -1)
    dir_min = torch.cat((dir_min, T1z), -1)

    # second direction (U!=0 or W!=0)
    T2x = (-V - s * torch.sqrt(torch.abs(V ** 2 - 4 * U * W) + 1e-10)) * grad[..., [2]]
    T2y = 2 * U * grad[..., [2]]
    T2z = (V + s * torch.sqrt(torch.abs(V ** 2 - 4 * U * W) + 1e-10)) * grad[..., [0]] - 2 * U * grad[..., [1]]
    dir_max = torch.cat((T2x, T2y), -1)
    dir_max = torch.cat((dir_max, T2z), -1)

    # first direction (U==0 and W==0)
    T1x_UW = torch.zeros_like(grad[..., [0]])
    T1y_UW = grad[..., [2]]
    T1z_UW = - grad[..., [1]]
    dir_min_UW = torch.cat((T1x_UW, T1y_UW), -1)
    dir_min_UW = torch.cat((dir_min_UW, T1z_UW), -1)

    # second direction (U==0 and W==0)
    T2x_UW = grad[..., [2]]
    T2y_UW = torch.zeros_like(grad[..., [0]])
    T2z_UW = - grad[..., [0]]
    dir_max_UW = torch.cat((T2x_UW, T2y_UW), -1)
    dir_max_UW = torch.cat((dir_max_UW, T2z_UW), -1)

    dir_min = torch.where(mask_3, dir_min, dir_min_UW)
    dir_max = torch.where(mask_3, dir_max, dir_max_UW)

    #computing the umbilical points
    # umbilical = torch.where(torch.abs(U)+torch.abs(V)+torch.abs(W)<1e-6, -1, 0)

    # normalizing the principal directions
    len_min=dir_min.norm(dim=-1).unsqueeze(-1)
    len_max=dir_max.norm(dim=-1).unsqueeze(-1)

    return dir_min/len_min, dir_max/len_max


def principal_curvature_parallel_surface(Kmin, Kmax, t):
    Kg = Kmin*Kmax
    Km = 0.5*(Kmin+Kmax)

    #curvatures of the parallel surface [manfredo, pg253]
    aux = np.ones_like(Kg) - 2.*t*Km + t*t*Kg
    aux[np.absolute(aux) < 0.0000001] = 0.0000001

    aux_zero = aux[np.absolute(aux) < 0.0000001]
    if aux_zero.size > 0:
        raise Exception('aux has zero members: ' + str(aux_zero))

    newKg = Kg/aux
    newKm = (Km-t*Kg)/aux

    A = np.sqrt(np.absolute(newKm**2 - newKg) + 0.00001)
    newKmax = newKm + A
    newKmin = newKm - A

    return newKmin, newKmax


def principal_curvature_region_detection(y, x):
    grad = gradient(y, x)
    hess = hessian(y, x)

    # principal curvatures
    min_curvature, max_curvature = principal_curvature(y, x, grad, hess)

    # Harris detector formula
    return min_curvature*max_curvature - 0.05*(min_curvature+max_curvature)**2
    # return min_curvature*max_curvature - 0.5*(min_curvature+max_curvature)**2


def umbilical_indicator(y, x):
    grad = gradient(y, x)
    hess = hessian(y, x)

    # principal curvatures
    min_curvature, max_curvature = principal_curvature(y, x, grad, hess)

    # Harris detector formula
    # return min_curvature*max_curvature - 0.05*(min_curvature+max_curvature)**2
    return 1-torch.abs(torch.tanh(min_curvature)-torch.tanh(max_curvature))


def tensor_curvature(y, x):
    grad = gradient(y, x)
    grad_norm = torch.norm(grad, dim=-1)
    unit_grad = grad.squeeze(-1)/grad_norm.unsqueeze(-1)

    T = -jacobian(unit_grad, x)[0]

    print(T)
    e, v = torch.eig(T, eigenvectors=True)

    print(e)
    print(v)

    return T


def gauss_bonnet_integral(grad,hess):
    Kg = gaussian_curvature(grad,hess).unsqueeze(-1)

    # remenber to restrict to the surface
    #Kg = torch.where(gt_sdf != -1, Kg, torch.zeros_like(Kg))

    aux = gradient.squeeze(-1)/torch.abs(gradient[:,:,0].unsqueeze(-1))

    Kg = Kg*(aux.norm(dim=-1).unsqueeze(-1))
    return torch.sum(Kg)/(Kg.shape[1]*0.5)


def hessian(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Hessian of y wrt x

    Parameters
    ----------
    y: torch.Tensor
        Shape [B, N, D_out], where B is the batch size (usually 1), N is the
        number of points, and D_out is the number of output channels of the
        network.

    x: torch.Tensor
        Shape [B, N, D_in],  where B is the batch size (usually 1), N is the
        number of points, and D_in is the number of input channels of the
        network.

    Returns
    -------
    h: torch.Tensor
        Shape: [B, N, D_out, D_in, D_in]
    """
    meta_batch_size, num_observations = y.shape[:2]
    grad_y = torch.ones_like(y[..., 0]).to(y.device)
    h = torch.zeros(meta_batch_size, num_observations, y.shape[-1], x.shape[-1], x.shape[-1]).to(y.device)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        dydx = grad(y[..., i], x, grad_y, create_graph=True)[0]

        # calculate hessian on y for each x value
        for j in range(x.shape[-1]):
            tmp = grad(dydx[..., j], x, grad_y, create_graph=True)
            h[..., i, j, :] = tmp[0].unsqueeze(1)[..., :]

    return h


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(
        y, [x], grad_outputs=grad_outputs, create_graph=True
    )[0]
    return grad


def jacobian(y: torch.Tensor, x: torch.Tensor):
    """Jacobian of y wrt x.

    Parameters
    ----------
    y: torch.Tensor
        A [B, N, A] tensor with the outputs of f(x)

    x: torch.Tensor
        A [B, N, D] tensor such that f(x) = y

    Returns
    -------
    jac: torch.Tensor
        A [B, N, A, D] tensor with the jacobian of y wrt x.

    status: int
        -1 if any NaN were introduced by the calculations, 0 otherwise
    """
    meta_batch_size, num_observations = y.shape[:2]
    jac = torch.zeros(meta_batch_size, num_observations, y.shape[-1], x.shape[-1]).to(y.device)  # (meta_batch_size*num_points, 2, 2)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        y_flat = y[..., i].view(-1, 1)
        jac[:, :, i, :] = grad(
            y_flat, x, torch.ones_like(y_flat), create_graph=True
        )[0]

    status = 0
    if torch.any(torch.isnan(jac)):
        status = -1

    return jac, status
