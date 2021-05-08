from numpy.lib.function_base import diff
import torch
from torch.autograd import grad
import itertools

import diff_operators

import numpy as np

class torus(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, coords):
        coords.requires_grad = True
        # Torus surface
        #f(x,y,z) = (sqrt(x**2 + y**2 -R))**2 + z**2 - r**2
        dist = (torch.sqrt(torch.abs(coords[...,0]**2 + coords[...,1]**2)+0.0000001) - 0.6)**2 + coords[...,2]**2 - 0.25**2
        return {"model_in": coords, "model_out": dist.unsqueeze(-1)}

class sdf_torus(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, coords):
        coords.requires_grad = True
        
        x = coords[...,0]
        y = coords[...,1]
        z = coords[...,2]
        
        tx = 0.6
        ty = 0.3

        qx = torch.sqrt(x**2+z**2)-tx
        qy = y
        dist = torch.sqrt(qx**2+qy**2)-ty

        return {"model_in": coords, "model_out": dist.unsqueeze(-1)}

class elipsoid(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, coords, t=15):
        coords.requires_grad = True

        #Elipsoid surface
        #f(x,y,z)= 3*x*x+25*y*y+z*z-0.5
        #grad(f) = [6x, 50y, 2z]
        #hess(f) = [[6,0,0],[0,50,0],[0,0,2]]
        dist = 3*coords[...,0]**2 + t*coords[...,1]**2 + coords[...,2]**2 - 0.5

        return {"model_in": coords, "model_out": dist.unsqueeze(-1)}

class octahedron(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, coords):
        coords.requires_grad = True

        #f(x,y,z)=x**4 + y**4 + z**4 + 6*x**2*y**2 + 6*y**2*z**2 + 6*z**2*x**2 − 1 
        dist =coords[...,0]**4 + coords[...,1]**4 + coords[...,2]**4 \
                               + 6*coords[...,0]**2*coords[...,1]**2 \
                               + 6*coords[...,1]**2*coords[...,2]**2 \
                               + 6*coords[...,2]**2*coords[...,0]**2 - 1

        return {"model_in": coords, "model_out": dist.unsqueeze(-1)}


class double_torus(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, coords):
        coords.requires_grad = True

        x = 2*coords[...,0]
        y = 2*coords[...,1]
        z = 2*coords[...,2]
    
        f = 2*y*(y**2-3*x**2)*(1-z**2)+(x**2+y**2)**2-(9*z**2-1)*(1-z**2)

        return {"model_in": coords, "model_out": f.unsqueeze(-1)}


