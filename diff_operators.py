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

    Km = 0.5*divergence(unit_grad, x)
    return Km

def principal_curvature(y, x, grad, hess):
    Kg = gaussian_curvature(grad,hess).unsqueeze(-1)
    Km = mean_curvature(y,x).squeeze(0)
    A = torch.sqrt(torch.abs(torch.pow(Km,2) - Kg) + 0.00001)
    Kmax = Km + A
    Kmin = Km - A

    #print(Kmax-Kmin)

    #return Kmin, Kmax
    return -Kmax, -Kmin

#Che, Wujun, Jean-Claude Paul, and Xiaopeng Zhang. 
#"Lines of curvature and umbilical points for implicit surfaces.
#" Computer Aided Geometric Design 24.7 (2007): 395-409.
def principal_directions(grad, hess):
    # Hz = grad[...,[2]].cpu().detach().numpy()
    # Hz0 = Hz[np.absolute(Hz)<0.00001]
    # if(Hz0.size > 0):
    #     print(Hz0)
    
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

    #first direction
    T1x = (-V + s*torch.sqrt(torch.abs(V**2-4*U*W)+1e-10))*grad[...,[2]]
    T1y = 2*U*grad[...,[2]]
    T1z = ( V - s*torch.sqrt(torch.abs(V**2-4*U*W)+1e-10))*grad[...,[0]] - 2*U*grad[...,[1]]
    T1 =  torch.cat((T1x, T1y), -1)
    T1 =  torch.cat((T1 , T1z), -1)

    #second direction
    T2x = (-V - s*torch.sqrt(torch.abs(V**2-4*U*W)+1e-10))*grad[...,[2]]
    T2y = 2*U*grad[...,[2]]
    T2z = ( V + s*torch.sqrt(torch.abs(V**2-4*U*W)+1e-10))*grad[...,[0]] - 2*U*grad[...,[1]]
    T2 =  torch.cat((T2x, T2y), -1)
    T2 =  torch.cat((T2 , T2z), -1)

    #computing the umbilical points
    umbilical = torch.where(torch.abs(U)+torch.abs(V)+torch.abs(W)<1e-6, -1, 0)
    T1 = torch.cat((T1,umbilical), -1)
    T2 = torch.cat((T2,umbilical), -1)

    return T1, T2

def principal_curvature_parallel_surface(Kmin, Kmax, t):
    Kg = Kmin*Kmax
    Km = 0.5*(Kmin+Kmax)

    #curvatures of the parallel surface [manfredo, pg253]
    aux = np.ones_like(Kg) - 2.*t*Km + t*t*Kg
    aux[np.absolute(aux)<0.0000001] = 0.0000001

    aux_zero = aux[np.absolute(aux)<0.0000001]
    if(aux_zero.size > 0):
        raise Exception('aux has zero members: ' + str(aux_zero))
    
    newKg = Kg/aux
    newKm = (Km-t*Kg)/aux

    A = np.sqrt(np.absolute(newKm**2 - newKg) + 0.00001)
    newKmax = newKm + A
    newKmin = newKm - A

    return newKmin, newKmax

def principal_curvature_region_detection(y,x):
    grad = gradient(y, x)
    hess = hessian(y, x)

    # principal curvatures
    min_curvature, max_curvature = principal_curvature(y, x, grad, hess)

    #Harris detector formula
    return min_curvature*max_curvature - 0.05*(min_curvature+max_curvature)**2
    #return min_curvature*max_curvature - 0.5*(min_curvature+max_curvature)**2

def umbilical_indicator(y,x):
    grad = gradient(y, x)
    hess = hessian(y, x)

    # principal curvatures
    min_curvature, max_curvature = principal_curvature(y, x, grad, hess)

    #Harris detector formula
    #return min_curvature*max_curvature - 0.05*(min_curvature+max_curvature)**2
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




