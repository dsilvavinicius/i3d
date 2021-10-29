import torch
import torch.nn.functional as F

import diff_operators
import modules

import trimesh

def sdf_constraint_on_surf(gt_sdf, pred_sdf):
   return torch.where(gt_sdf == 0, pred_sdf ** 2, torch.zeros_like(pred_sdf))

def sdf_constraint_off_surf(gt_sdf, pred_sdf):
   return torch.where(gt_sdf != 0, (gt_sdf - pred_sdf) ** 2, torch.zeros_like(pred_sdf))

def vector_aligment_on_surf(gt_sdf, gt_vectors, pred_vectors):
   return torch.where(gt_sdf == 0, 1 - F.cosine_similarity(pred_vectors, gt_vectors, dim=-1)[..., None], torch.zeros_like(gt_sdf))

def direction_aligment_on_surf(gt_sdf, gt_dirs, pred_dirs):
   return torch.where(gt_sdf == 0, 1 - (F.cosine_similarity(pred_dirs, gt_dirs, dim=-1)[..., None])**2, torch.zeros_like(gt_sdf))
    
def eikonal_constraint(gradient):
   return ((gradient.norm(dim=-1) - 1.) ** 2).unsqueeze(-1)


def true_sdf(model_output, gt):
    '''Uses true SDF value off surface.
    x: batch of input coordinates
    y: usually the output of the trial_soln function
    '''
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    gradient = diff_operators.gradient(pred_sdf, coords)
    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    return {'sdf_on_surf': sdf_constraint_on_surf(gt_sdf, pred_sdf).mean() * 3e3,
            'sdf_off_surf': sdf_constraint_off_surf(gt_sdf, pred_sdf).mean() * 2e2,
            'normal_constraint': vector_aligment_on_surf(gt_sdf, gt_normals, gradient).mean() *1e2 ,#* 1e1,
            'grad_constraint': eikonal_constraint(gradient).mean() * 5e1}#1e1}

def loss_curvatures(model_output, gt):
    '''Uses true SDF value off surface and tries to fit gaussian curvatures on
    the 0 level-set.

    x: batch of input coordinates
    y: usually the output of the trial_soln function
    '''
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']
    gt_min_curvature = gt["min_curvatures"]
    gt_max_curvature = gt["max_curvatures"]
    gt_dirs = gt["max_principal_directions"]

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    gradient = diff_operators.gradient(pred_sdf, coords)
    hessian = diff_operators.hessian(pred_sdf, coords)
    pred_dirs = diff_operators.principal_directions(gradient, hessian[0])

    dirs_constraint = direction_aligment_on_surf(gt_sdf, gt_dirs, pred_dirs[0][...,0:3])

    aux_dirs_constraint = torch.where(gt_sdf == 0, F.cosine_similarity(pred_dirs[0][...,0:3], gt_normals, dim=-1)[..., None]**2, torch.zeros_like(gt_sdf))

    dirs_constraint = dirs_constraint + 0.1*aux_dirs_constraint

   #removing umbilical points of the pred sdf
    dirs_constraint = torch.where(pred_dirs[0][...,3].unsqueeze(-1) == 0, dirs_constraint, torch.zeros_like(dirs_constraint))

    #removing problematic curvatures and planar points
    planar_curvature = 0.5*torch.abs(gt_min_curvature-gt_max_curvature)
    dirs_constraint = torch.where(planar_curvature > 2.5  , dirs_constraint, torch.zeros_like(dirs_constraint))
    dirs_constraint = torch.where(planar_curvature < 5000, dirs_constraint, torch.zeros_like(dirs_constraint))

    return {'sdf_on_surf': sdf_constraint_on_surf(gt_sdf, pred_sdf).mean() * 3e3,
            'sdf_off_surf': sdf_constraint_off_surf(gt_sdf, pred_sdf).mean() * 2e2,
            'normal_constraint': vector_aligment_on_surf(gt_sdf, gt_normals, gradient).mean() *1e2,#* 1e1,
            'grad_constraint': eikonal_constraint(gradient).mean() * 5e1,
            'dirs_constraint': dirs_constraint.mean()
            }


def true_sdf_curvature(model_output, gt):
    '''Uses true SDF value off surface and tries to fit gaussian curvatures on
    the 0 level-set.

    x: batch of input coordinates
    y: usually the output of the trial_soln function
    '''
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']
    gt_curvature = gt["curvature"]

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    gradient = diff_operators.gradient(pred_sdf, coords)

   # mean curvature
    pred_curvature = diff_operators.mean_curvature(pred_sdf, coords)
    curvature_diff = torch.tanh(100*pred_curvature) - torch.tanh(100*gt_curvature)

    #consider the curature constraint only on the surface
    curv_constraint = torch.where(gt_sdf == 0, curvature_diff ** 2, torch.zeros_like(pred_curvature))
    #remove problematic curvatures and planar points
    curv_constraint = torch.where(torch.abs(gt_curvature) < 5000, curv_constraint, torch.zeros_like(pred_curvature))
    curv_constraint = torch.where(torch.abs(gt_curvature) > 10, curv_constraint, torch.zeros_like(pred_curvature))

    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    return {'sdf_on_surf': sdf_constraint_on_surf(gt_sdf, pred_sdf).mean() * 3e3,
            'sdf_off_surf': sdf_constraint_off_surf(gt_sdf, pred_sdf).mean() * 2e2,
            'normal_constraint': vector_aligment_on_surf(gt_sdf, gt_normals, gradient).mean() *1e2 ,#* 1e1,
            'grad_constraint': eikonal_constraint(gradient).mean() * 5e1,
            'curv_constraint': curv_constraint.mean() * 5 }