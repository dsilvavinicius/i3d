import torch
from torch.autograd import grad
import itertools

import numpy as np

def gaussian_curvature(grad, hess):
    ''' curvature of a implicit surface (https://en.wikipedia.org/wiki/Gaussian_curvature#Alternative_formulas).
    '''
    if(hess[1] == -1):
        raise Exception('Hessian has NaN members: ' + str(hess[0]))
    
    # Append gradients to the last columns of the hessians.
    grad5d = torch.unsqueeze(grad, 2)
    grad5d = torch.unsqueeze(grad5d, -1)
    F = torch.cat((hess[0], grad5d), -1)
    
    # Append gradients (with and additional 0 at the last coord) to the last lines of the hessians.
    hess_size = hess[0].size()
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

    Km = -0.5*divergence(unit_grad, x)
    return Km

# curvature using numpy
def np_mean_curvature(y, x):
    grad = np_gradient(y, x)
    grad_norm = np.norm(grad, dim=-1)
    unit_grad = grad.squeeze(-1)/grad_norm.unsqueeze(-1)

    Km = -0.5*divergence(unit_grad, x)
    return Km    

def principal_curvature(y, x, grad, hess):
    Kg = gaussian_curvature(grad,hess).unsqueeze(-1)
    Km = mean_curvature(y,x).squeeze(0)
    A = torch.sqrt(torch.abs(torch.pow(Km,2) - Kg) + 0.0001)
    Kmax = Km + A
    Kmin = Km - A
    return Kmin, Kmax

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
 

def hessian(y, x):
    ''' hessian of y wrt x
    y: shape (meta_batch_size, num_observations, channels)
    x: shape (meta_batch_size, num_observations, 2)
    '''
    meta_batch_size, num_observations = y.shape[:2]
    grad_y = torch.ones_like(y[..., 0]).to(y.device)
    h = torch.zeros(meta_batch_size, num_observations, y.shape[-1], x.shape[-1], x.shape[-1]).to(y.device)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        dydx = grad(y[..., i], x, grad_y, create_graph=True)[0]

        # calculate hessian on y for each x value
        for j in range(x.shape[-1]):
            h[..., i, j, :] = grad(dydx[..., j], x, grad_y, create_graph=True)[0][..., :]

    status = 0
    if torch.any(torch.isnan(h)):
        status = -1
    return h, status


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
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def np_gradient(y, x, grad_outputs=None):
    y = y.cpu().detach().numpy()
    x = x.cpu().detach().numpy()
    if grad_outputs is None:
        grad_outputs = np.ones_like(y)

    print(y.shape)
    print(x.shape)
    grad = np.gradient(y, x)[0]
    return grad


def jacobian(y, x):
    ''' jacobian of y wrt x '''
    meta_batch_size, num_observations = y.shape[:2]
    jac = torch.zeros(meta_batch_size, num_observations, y.shape[-1], x.shape[-1]).to(y.device) # (meta_batch_size*num_points, 2, 2)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        y_flat = y[...,i].view(-1, 1)
        jac[:, :, i, :] = grad(y_flat, x, torch.ones_like(y_flat), create_graph=True)[0]

    status = 0
    if torch.any(torch.isnan(jac)):
        status = -1

    return jac, status




