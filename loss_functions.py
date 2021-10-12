import torch
import torch.nn.functional as F

import diff_operators
import modules


def sdf_original(model_output, gt):
    '''
       x: batch of input coordinates
       y: usually the output of the trial_soln function
       '''
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']

    coords   = model_output['model_in']
    pred_sdf = model_output['model_out']

    gradient = diff_operators.gradient(pred_sdf, coords)
    grad_norm = gradient.norm(dim=-1)
    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    sdf_constraint    = torch.where(gt_sdf != -1, (gt_sdf - pred_sdf)**2, torch.zeros_like(pred_sdf))
    inter_constraint  = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
    normal_constraint = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None], torch.zeros_like(gradient[..., :1]))
    grad_constraint   = torch.where(gt_sdf != -1, (grad_norm - 1.)**2, torch.zeros_like(pred_sdf))
    #grad_constraint   = (gradient.norm(dim=-1) - 1.)**2
    
    # Exp      # Lapl
    # -----------------
    return {'sdf': torch.abs(sdf_constraint).mean() * 3e3, #3e3,  # 1e4      # 3e3
            'inter': inter_constraint.mean() * 1e1,  # 1e2                   # 1e3
            'normal_constraint': normal_constraint.mean() * 1e2,  # 1e2
            'grad_constraint': grad_constraint.mean() * 1e1}


def sdf(model_output, gt):
    '''
       x: batch of input coordinates
       y: usually the output of the trial_soln function
       '''
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']

    coords   = model_output['model_in']
    pred_sdf = model_output['model_out']

    gradient = diff_operators.gradient(pred_sdf, coords)
    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    sdf_constraint    = (gt_sdf - pred_sdf)**2
    normal_constraint = torch.where(gt_sdf == 0, 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None], torch.zeros_like(gradient[..., :1]))
    grad_constraint   = (gradient.norm(dim=-1) - 1.)**2
    
    # Exp      # Lapl
    # -----------------
    return {'sdf': torch.abs(sdf_constraint).mean() * 1e1, #3e3,  # 1e4      # 3e3
            'normal_constraint': normal_constraint.mean() ,  # 1e2
            'grad_constraint': grad_constraint.mean() }            

def sdf_original_on_surface(model_output, gt):
    '''
       x: batch of input coordinates
       y: usually the output of the trial_soln function
       '''
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']

    coords   = model_output['model_in']
    pred_sdf = model_output['model_out']
 
    gradient = diff_operators.gradient(pred_sdf, coords)
    #grad_norm = gradient.norm(dim=-1)
    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    sdf_constraint    = torch.where(gt_sdf != -1, (gt_sdf - pred_sdf)**2, torch.zeros_like(pred_sdf))
    #inter_constraint  = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
    normal_constraint = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None], torch.zeros_like(gradient[..., :1]))
    #grad_constraint   = torch.where(gt_sdf != -1, (grad_norm - 1.)**2, torch.zeros_like(pred_sdf))
    grad_constraint   = (gradient.norm(dim=-1) - 1.)**2
    
    # Exp      # Lapl
    # -----------------
    return {'sdf': torch.abs(sdf_constraint).mean() * 3e4, #3e3,  # 1e4      # 3e3
   #         'inter': inter_constraint.mean() * 1e1,  # 1e2                   # 1e3
            'normal_constraint': normal_constraint.mean() * 1e3,  # 1e2
            'grad_constraint': grad_constraint.mean() * 1e1}


def sdf_principal_curvature_segmentation(model_output, gt):
    '''
       x: batch of input coordinates
       y: usually the output of the trial_soln function
       '''
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']

    coords   = model_output['model_in']
    pred_sdf = model_output['model_out']

    gradient = diff_operators.gradient(pred_sdf, coords)
    grad_norm = gradient.norm(dim=-1)
    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    sdf_constraint    = torch.where(gt_sdf != -1, (gt_sdf - pred_sdf)**2, torch.zeros_like(pred_sdf))
    inter_constraint  = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
    normal_constraint = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None], torch.zeros_like(gradient[..., :1]))
    grad_constraint   = torch.where(gt_sdf != -1, (grad_norm - 1.)**2, torch.zeros_like(pred_sdf))
    #grad_constraint   = (gradient.norm(dim=-1) - 1.)**2
    
    planar_weight = 0.2
    edge_weight = 0.4
    corner_weight = 0.4
    harris_detector = diff_operators.principal_curvature_region_detection(pred_sdf, coords)
    
    H1 = (planar_weight-edge_weight)*torch.sigmoid(harris_detector) + edge_weight*torch.ones_like(harris_detector)
    H  = (corner_weight*torch.ones_like(harris_detector) - H1)*torch.sigmoid(harris_detector-50.*torch.ones_like(harris_detector)) + H1

    sdf_constraint = H*sdf_constraint
    normal_constraint = H*normal_constraint
    #sdf_constraint = torch.where(torch.abs(harris_detector) < 10., planar_weight*sdf_constraint, corner_weight*sdf_constraint) #planar region
    #sdf_constraint = torch.where(harris_detector < 0.0, edge_weight*sdf_constraint, sdf_constraint)

    # Exp      # Lapl
    # -----------------
    return {'sdf': sdf_constraint.mean() * 3e4, #3e3,  # 1e4      # 3e3
            'inter': inter_constraint.mean() * 1e1,  # 1e2                   # 1e3
            'normal_constraint': normal_constraint.mean() * 1e3,  # 1e2
            'grad_constraint': grad_constraint.mean() * 1e1}


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
    sdf_constraint_on_surf = torch.where(gt_sdf == 0, pred_sdf ** 2, torch.zeros_like(pred_sdf))
    sdf_constraint_off_surf = torch.where(gt_sdf != 0, (gt_sdf - pred_sdf) ** 2, torch.zeros_like(pred_sdf))
#    sdf_constraint = (gt_sdf - pred_sdf)**2
    normal_constraint = torch.where(gt_sdf == 0, 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None], torch.zeros_like(gradient[..., :1]))
    grad_constraint = (gradient.norm(dim=-1) - 1.) ** 2

    # Exp      # Lapl
    # -----------------
    return {'sdf_on_surf': (sdf_constraint_on_surf ** 2).mean() * 3e3,
            'sdf_off_surf': sdf_constraint_off_surf.mean() * 2e2,
            'normal_constraint': normal_constraint.mean() * 1e1,
            'grad_constraint': grad_constraint.mean() * 1e1}


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
    hessian = diff_operators.hessian(pred_sdf, coords)

    # gaussian curvature
    pred_curvature = diff_operators.gaussian_curvature(gradient, hessian).unsqueeze(-1)
    curvature_diff = torch.tanh(pred_curvature) - torch.tanh(gt_curvature)

    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    sdf_constraint_on_surf = torch.where(gt_sdf == 0, pred_sdf ** 2, torch.zeros_like(pred_sdf))
    sdf_constraint_off_surf = torch.where(gt_sdf != 0, (gt_sdf - pred_sdf) ** 2, torch.zeros_like(pred_sdf))
#    sdf_constraint = (gt_sdf - pred_sdf)**2
    normal_constraint = torch.where(gt_sdf == 0, 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None], torch.zeros_like(gradient[..., :1]))
    grad_constraint = (gradient.norm(dim=-1) - 1.) ** 2
    curv_constraint = torch.where(
        gt_sdf == 0,
        curvature_diff ** 2,
        torch.zeros_like(pred_curvature)
    )

    # Exp      # Lapl
    # -----------------
    return {'sdf_on_surf': (sdf_constraint_on_surf ** 2).mean() * 3e3,
            'sdf_off_surf': sdf_constraint_off_surf.mean() * 2e2,
            'normal_constraint': normal_constraint.mean() * 1e1,
            'grad_constraint': grad_constraint.mean() * 1e1,
            "curv_constraint": curv_constraint.mean()}


def sdf_mean_curvature(model_output, gt):
    '''
       x: batch of input coordinates
       y: usually the output of the trial_soln function
       '''
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']

    gt_min_curvature = gt['min_curvature']
    gt_max_curvature = gt['max_curvature']

    gt_curvature = 0.5*(gt_min_curvature + gt_max_curvature)
    
    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    gradient = diff_operators.gradient(pred_sdf, coords)
 
    # mean curvature
    pred_curvature = diff_operators.mean_curvature(pred_sdf, coords)
    curvature_diff = torch.where(gt_sdf != -1, torch.tanh(1000*pred_curvature) - torch.tanh(1000*gt_curvature), torch.zeros_like(pred_curvature))

    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    sdf_constraint = torch.where(gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf))
    inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
    normal_constraint = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None], torch.zeros_like(gradient[..., :1]))
    grad_constraint = (gradient.norm(dim=-1) - 1)**2
    curv_constraint = torch.where(gt_sdf != -1, torch.pow(curvature_diff, 2), torch.zeros_like(pred_curvature))

    abs_curvature = torch.abs(gt_curvature)
    normal_constraint = torch.where(abs_curvature> 5., normal_constraint, torch.zeros_like(gradient[..., :1]))
    curv_constraint = torch.where(abs_curvature> 10., curv_constraint, torch.zeros_like(pred_curvature))

    # Exp      # Lapl
    # -----------------
    return {'sdf': torch.abs(sdf_constraint).mean() * 3e3,  # 1e4      # 3e3
            'inter': inter_constraint.mean() * 1e2,  # 1e2                   # 1e3
            'normal_constraint': normal_constraint.mean() * 1e2,  # 1e2
            'grad_constraint': grad_constraint.mean() * 5e1,
            'curv_constraint': curv_constraint.mean() }


def sdf_gaussian_curvature(model_output, gt):
    '''
       x: batch of input coordinates
       y: usually the output of the trial_soln function
       '''
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']

    gt_min_curvature = gt['min_curvature']
    gt_max_curvature = gt['max_curvature']

    gt_curvature = gt_min_curvature*gt_max_curvature
    
    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    gradient = diff_operators.gradient(pred_sdf, coords)
    hessian = diff_operators.hessian(pred_sdf, coords)

    # gaussian curvature
    pred_curvature = diff_operators.gaussian_curvature(gradient, hessian).unsqueeze(-1)
    curvature_diff = torch.tanh(0.01*pred_curvature) - torch.tanh(0.01*gt_curvature)

    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    sdf_constraint = torch.where(gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf))
    inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
    normal_constraint = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None], torch.zeros_like(gradient[..., :1]))
    grad_constraint = (gradient.norm(dim=-1) - 1)**2
    curv_constraint = torch.where(gt_sdf != -1, torch.pow(curvature_diff, 2), torch.zeros_like(pred_curvature))

    abs_curvature = torch.abs(gt_curvature)
    normal_constraint = torch.where(abs_curvature> 5., normal_constraint, torch.zeros_like(gradient[..., :1]))
    curv_constraint = torch.where(abs_curvature > 10., curv_constraint, torch.zeros_like(pred_curvature))

    # Exp      # Lapl
    # -----------------
    return {'sdf': torch.abs(sdf_constraint).mean() * 3e3,  # 1e4      # 3e3
            'inter': inter_constraint.mean() * 1e2,  # 1e2                   # 1e3
            'normal_constraint': normal_constraint.mean() * 1e2,  # 1e2
            'grad_constraint': grad_constraint.mean() * 5e1,
            'curv_constraint': curv_constraint.mean() }


def sdf_principal_curvatures(model_output, gt):
    '''
       x: batch of input coordinates
       y: usually the output of the trial_soln function
       '''
    #th.autograd.set_detect_anomaly(True)

    gt_sdf = gt['sdf']
    gt_normals = gt['normals']

    gt_min_curvature = gt['min_curvature']
    gt_max_curvature = gt['max_curvature']
    
    coords = model_output['model_in']
    pred_sdf = model_output['model_out']
    
    gradient = diff_operators.gradient(pred_sdf, coords)
    hessian = diff_operators.hessian(pred_sdf, coords)

    # principal curvatures
    pred_min_curvature, pred_max_curvature = diff_operators.principal_curvature(pred_sdf, coords, gradient, hessian)
    curv_constraint = (torch.tanh(pred_min_curvature) - torch.tanh(gt_min_curvature))**2 + (torch.tanh(pred_max_curvature) - torch.tanh(gt_max_curvature))**2
    
    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    sdf_constraint = torch.where(gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf))
    inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
    normal_constraint = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None], torch.zeros_like(gradient[..., :1]))
    grad_constraint = torch.where(gt_sdf != -1, (gradient.norm(dim=-1) - 1.)**2, torch.zeros_like(pred_sdf))
    curv_constraint = torch.where(gt_sdf != -1, curv_constraint, torch.zeros_like(curv_constraint))
    
    #removing regions with lower curvatures
    curv_constraint =   torch.where(torch.abs(gt_min_curvature) > 1., curv_constraint, torch.zeros_like(curv_constraint))
    curv_constraint =   torch.where(torch.abs(gt_max_curvature) > 1., curv_constraint, torch.zeros_like(curv_constraint))

    #removing points close to be umbilicals
    #curv_constraint =   torch.where(torch.abs(gt_min_curvature-gt_max_curvature) > 0.0005, curv_constraint, torch.zeros_like(curv_constraint))

    # Exp      # Lapl
    # -----------------
    return {
            'sdf': torch.abs(sdf_constraint).mean() * 1e3,  # 1e4      # 3e3
            'inter': inter_constraint.mean() * 1e2,  # 1e2                   # 1e3
            'normal_constraint': normal_constraint.mean() * 1e2,  # 1e2
            'grad_constraint': grad_constraint.mean() * 1e1,
            'curv_constraint': curv_constraint.mean()
            }


def sdf_tensor_curvature(model_output, gt):
    '''
       x: batch of input coordinates
       y: usually the output of the trial_soln function
       '''
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']

    gt_min_curvature = gt['min_curvature']
    gt_max_curvature = gt['max_curvature']

    gt_curvature = 0.5*(gt_min_curvature + gt_max_curvature)
    
    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    tensor =diff_operators.tensor_curvature(pred_sdf, coords)

    gradient = diff_operators.gradient(pred_sdf, coords)
 
    # mean curvature
    pred_curvature = diff_operators.mean_curvature(pred_sdf, coords)
    curvature_diff = torch.where(gt_sdf != -1, torch.tanh(1000*pred_curvature) - torch.tanh(1000*gt_curvature), torch.zeros_like(pred_curvature))

    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    sdf_constraint = torch.where(gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf))
    inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
    normal_constraint = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None], torch.zeros_like(gradient[..., :1]))
    grad_constraint = (gradient.norm(dim=-1) - 1)**2
    curv_constraint = torch.where(gt_sdf != -1, torch.pow(curvature_diff, 2), torch.zeros_like(pred_curvature))

    abs_curvature = torch.abs(gt_curvature)
    normal_constraint = torch.where(abs_curvature> 5., normal_constraint, torch.zeros_like(gradient[..., :1]))
    curv_constraint = torch.where(abs_curvature> 10., curv_constraint, torch.zeros_like(pred_curvature))

    # Exp      # Lapl
    # -----------------
    return {'sdf': torch.abs(sdf_constraint).mean() * 3e3,  # 1e4      # 3e3
            'inter': inter_constraint.mean() * 1e2,  # 1e2                   # 1e3
            'normal_constraint': normal_constraint.mean() * 1e2,  # 1e2
            'grad_constraint': grad_constraint.mean() * 5e1,
            'curv_constraint': curv_constraint.mean() }


def sdf_principal_directions(model_output, gt):
    '''
       x: batch of input coordinates
       y: usually the output of the trial_soln function
       '''
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']
    gt_dirs = gt['principal_directions']

    coords   = model_output['model_in']
    pred_sdf = model_output['model_out']

    gradient = diff_operators.gradient(pred_sdf, coords)
    grad_norm = gradient.norm(dim=-1)
    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    sdf_constraint    = torch.where(gt_sdf != -1, (gt_sdf - pred_sdf)**2, torch.zeros_like(pred_sdf))
    inter_constraint  = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
    normal_constraint = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None], torch.zeros_like(gradient[..., :1]))
    grad_constraint   = torch.where(gt_sdf != -1, (grad_norm - 1.)**2, torch.zeros_like(pred_sdf))
    #grad_constraint   = (gradient.norm(dim=-1) - 1.)**2
    #grad_constraint = torch.abs(gradient.norm(dim=-1) - 1)

    #hessian = diff_operators.hessian(pred_sdf, coords)[0]
    #pred_dirs = diff_operators.principal_directions(gradient, hessian)
    #dirs_constraint = torch.where(gt_sdf != -1, 1 - torch.abs(F.cosine_similarity(pred_dirs, gt_dirs, dim=-1)[..., None]), torch.zeros_like(gradient[..., :1]))
    #dirs_constraint = torch.where(gt_sdf != -1, F.cosine_similarity(gradient, gt_dirs, dim=-1)[..., None]**2, torch.zeros_like(gradient[..., :1]))
    
    
    #test = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(pred_dirs, gradient, dim=-1)[..., None], torch.zeros_like(gradient[..., :1]))
    

    # Exp      # Lapl
    # -----------------
    return {'sdf': torch.abs(sdf_constraint).mean() * 3e3, #3e3,  # 1e4      # 3e3
            'inter': inter_constraint.mean() * 1e2,  # 1e2                   # 1e3
            'normal_constraint': normal_constraint.mean() * 1e2,  # 1e2
            'grad_constraint': grad_constraint.mean() * 1e1#,  # 1e2
            #'dirs_constraint': dirs_constraint.mean() * 1e1
            }


def implicit_function(model_output, gt):
    '''
       x: batch of input coordinates
       y: usually the output of the trial_soln function
       '''
    gt_sdf = gt['sdf']
    #gt_normals = gt['normals']
    #gt_dirs = gt['principal_directions']
    #gt_min_curv = gt['min_curvature']
    #gt_max_curv = gt['max_curvature']

    coords   = model_output['model_in']
    pred_sdf = model_output['model_out']

    #gradient = diff_operators.gradient(pred_sdf, coords)
    #hessian  = diff_operators.hessian(pred_sdf, coords)
    #pred_dirs = diff_operators.principal_directions(gradient,hessian[0])[0]
    #pred_min_curv, pred_max_curv = diff_operators.principal_curvature(pred_sdf,coords,gradient,hessian)   
    #pred_min_curv = pred_min_curv.unsqueeze(0)
    #pred_max_curv = pred_max_curv.unsqueeze(0)

    # print(gt_min_curv)
    # print(gt_max_curv)

    # print(pred_min_curv)
    # print(pred_max_curv)

    sdf_constraint    = torch.abs(gt_sdf - pred_sdf)
    #normal_constraint =  1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None]
    #grad_constraint   = (gt_normals-gradient).norm(dim=-1).unsqueeze(-1)
    #dirs_constraint =  1 - torch.abs(F.cosine_similarity(pred_dirs, gt_dirs, dim=-1)[..., None])
    #curv_constraint = (gt_min_curv-pred_min_curv)**2 + (gt_max_curv-pred_max_curv)**2

    # Exp      # Lapl
    # -----------------
    return {'sdf'              : sdf_constraint.mean() * 1e1 #,
            #'normal_constraint': normal_constraint.mean(),# *1
            #'grad_constraint'  : grad_constraint.mean() * 0.1#, 
            #'dirs_constraint'  : dirs_constraint.mean()*1e3,# *1 
            #'curv_constraint'  : curv_constraint.mean() *0.001 
            }


def implicit_function_4D(model_output, gt):
    '''
       x: batch of input coordinates
       y: usually the output of the trial_soln function
       '''
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']
    
    coords   = model_output['model_in']
    pred_sdf = model_output['model_out']

    gradient = diff_operators.gradient(pred_sdf, coords)[:,:,0:3]
    
    sdf_constraint    = (gt_sdf - pred_sdf)**2
    normal_constraint =  1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None]
    grad_constraint   = (gt_normals-gradient).norm(dim=-1).unsqueeze(-1)
    
    # Exp      # Lapl
    # -----------------
    return {'sdf'              : sdf_constraint.mean() * 2e1 ,
            'normal_constraint': normal_constraint.mean(),# *1
            'grad_constraint'  : grad_constraint.mean() * 0.1#, 
            }
