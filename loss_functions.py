import torch
from torch.functional import norm
import torch.nn.functional as F

import diff_operators
import modules

#import torch as th

def image_mse(mask, model_output, gt):
    if mask is None:
        return {'img_loss': ((model_output['model_out'] - gt['img']) ** 2).mean()}
    else:
        return {'img_loss': (mask * (model_output['model_out'] - gt['img']) ** 2).mean()}


def image_l1(mask, model_output, gt):
    if mask is None:
        return {'img_loss': torch.abs(model_output['model_out'] - gt['img']).mean()}
    else:
        return {'img_loss': (mask * torch.abs(model_output['model_out'] - gt['img'])).mean()}


def image_mse_TV_prior(mask, k1, model, model_output, gt):
    coords_rand = 2 * (torch.rand((model_output['model_in'].shape[0],
                                   model_output['model_in'].shape[1] // 2,
                                   model_output['model_in'].shape[2])).cuda() - 0.5)
    rand_input = {'coords': coords_rand}
    rand_output = model(rand_input)

    if mask is None:
        return {'img_loss': ((model_output['model_out'] - gt['img']) ** 2).mean(),
                'prior_loss': k1 * (torch.abs(diff_operators.gradient(
                    rand_output['model_out'], rand_output['model_in']))).mean()}
    else:
        return {'img_loss': (mask * (model_output['model_out'] - gt['img']) ** 2).mean(),
                'prior_loss': k1 * (torch.abs(diff_operators.gradient(
                    rand_output['model_out'], rand_output['model_in']))).mean()}


def image_mse_FH_prior(mask, k1, model, model_output, gt):
    coords_rand = 2 * (torch.rand((model_output['model_in'].shape[0],
                                   model_output['model_in'].shape[1] // 2,
                                   model_output['model_in'].shape[2])).cuda() - 0.5)
    rand_input = {'coords': coords_rand}
    rand_output = model(rand_input)

    img_hessian, status = diff_operators.hessian(rand_output['model_out'],
                                                 rand_output['model_in'])
    img_hessian = img_hessian.view(*img_hessian.shape[0:2], -1)
    hessian_norm = img_hessian.norm(dim=-1, keepdim=True)

    if mask is None:
        return {'img_loss': ((model_output['model_out'] - gt['img']) ** 2).mean(),
                'prior_loss': k1 * (torch.abs(hessian_norm)).mean()}
    else:
        return {'img_loss': (mask * (model_output['model_out'] - gt['img']) ** 2).mean(),
                'prior_loss': k1 * (torch.abs(hessian_norm)).mean()}


def latent_loss(model_output):
    return torch.mean(model_output['latent_vec'] ** 2)


def hypo_weight_loss(model_output):
    weight_sum = 0
    total_weights = 0

    for weight in model_output['hypo_params'].values():
        weight_sum += torch.sum(weight ** 2)
        total_weights += weight.numel()

    return weight_sum * (1 / total_weights)


def image_hypernetwork_loss(mask, kl, fw, model_output, gt):
    return {'img_loss': image_mse(mask, model_output, gt)['img_loss'],
            'latent_loss': kl * latent_loss(model_output),
            'hypo_weight_loss': fw * hypo_weight_loss(model_output)}


def function_mse(model_output, gt):
    return {'func_loss': ((model_output['model_out'] - gt['func']) ** 2).mean()}


def gradients_mse(model_output, gt):
    # compute gradients on the model
    gradients = diff_operators.gradient(model_output['model_out'], model_output['model_in'])
    # compare them with the ground-truth
    gradients_loss = torch.mean((gradients - gt['gradients']).pow(2).sum(-1))
    return {'gradients_loss': gradients_loss}


def gradients_color_mse(model_output, gt):
    # compute gradients on the model
    gradients_r = diff_operators.gradient(model_output['model_out'][..., 0], model_output['model_in'])
    gradients_g = diff_operators.gradient(model_output['model_out'][..., 1], model_output['model_in'])
    gradients_b = diff_operators.gradient(model_output['model_out'][..., 2], model_output['model_in'])
    gradients = torch.cat((gradients_r, gradients_g, gradients_b), dim=-1)
    # compare them with the ground-truth
    weights = torch.tensor([1e1, 1e1, 1., 1., 1e1, 1e1]).cuda()
    gradients_loss = torch.mean((weights * (gradients[0:2] - gt['gradients']).pow(2)).sum(-1))
    return {'gradients_loss': gradients_loss}


def laplace_mse(model_output, gt):
    # compute laplacian on the model
    laplace = diff_operators.laplace(model_output['model_out'], model_output['model_in'])
    # compare them with the ground truth
    laplace_loss = torch.mean((laplace - gt['laplace']) ** 2)
    return {'laplace_loss': laplace_loss}


def wave_pml(model_output, gt):
    source_boundary_values = gt['source_boundary_values']
    x = model_output['model_in']  # (meta_batch_size, num_points, 3)
    y = model_output['model_out']  # (meta_batch_size, num_points, 1)
    squared_slowness = gt['squared_slowness']
    dirichlet_mask = gt['dirichlet_mask']
    batch_size = x.shape[1]

    du, status = diff_operators.jacobian(y, x)
    dudt = du[..., 0]

    if torch.all(dirichlet_mask):
        diff_constraint_hom = torch.Tensor([0])
    else:
        hess, status = diff_operators.jacobian(du[..., 0, :], x)
        lap = hess[..., 1, 1, None] + hess[..., 2, 2, None]
        dudt2 = hess[..., 0, 0, None]
        diff_constraint_hom = dudt2 - 1 / squared_slowness * lap

    dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]
    neumann = dudt[dirichlet_mask]

    return {'dirichlet': torch.abs(dirichlet).sum() * batch_size / 1e1,
            'neumann': torch.abs(neumann).sum() * batch_size / 1e2,
            'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}


def helmholtz_pml(model_output, gt):
    source_boundary_values = gt['source_boundary_values']

    if 'rec_boundary_values' in gt:
        rec_boundary_values = gt['rec_boundary_values']

    wavenumber = gt['wavenumber'].float()
    x = model_output['model_in']  # (meta_batch_size, num_points, 2)
    y = model_output['model_out']  # (meta_batch_size, num_points, 2)
    squared_slowness = gt['squared_slowness'].repeat(1, 1, y.shape[-1] // 2)
    batch_size = x.shape[1]

    full_waveform_inversion = False
    if 'pretrain' in gt:
        pred_squared_slowness = y[:, :, -1] + 1.
        if torch.all(gt['pretrain'] == -1):
            full_waveform_inversion = True
            pred_squared_slowness = torch.clamp(y[:, :, -1], min=-0.999) + 1.
            squared_slowness_init = torch.stack((torch.ones_like(pred_squared_slowness),
                                                 torch.zeros_like(pred_squared_slowness)), dim=-1)
            squared_slowness = torch.stack((pred_squared_slowness, torch.zeros_like(pred_squared_slowness)), dim=-1)
            squared_slowness = torch.where((torch.abs(x[..., 0, None]) > 0.75) | (torch.abs(x[..., 1, None]) > 0.75),
                                           squared_slowness_init, squared_slowness)
        y = y[:, :, :-1]

    du, status = diff_operators.jacobian(y, x)
    dudx1 = du[..., 0]
    dudx2 = du[..., 1]

    a0 = 5.0

    # let pml extend from -1. to -1 + Lpml and 1 - Lpml to 1.0
    Lpml = 0.5
    dist_west = -torch.clamp(x[..., 0] + (1.0 - Lpml), max=0)
    dist_east = torch.clamp(x[..., 0] - (1.0 - Lpml), min=0)
    dist_south = -torch.clamp(x[..., 1] + (1.0 - Lpml), max=0)
    dist_north = torch.clamp(x[..., 1] - (1.0 - Lpml), min=0)

    sx = wavenumber * a0 * ((dist_west / Lpml) ** 2 + (dist_east / Lpml) ** 2)[..., None]
    sy = wavenumber * a0 * ((dist_north / Lpml) ** 2 + (dist_south / Lpml) ** 2)[..., None]

    ex = torch.cat((torch.ones_like(sx), -sx / wavenumber), dim=-1)
    ey = torch.cat((torch.ones_like(sy), -sy / wavenumber), dim=-1)

    A = modules.compl_div(ey, ex).repeat(1, 1, dudx1.shape[-1] // 2)
    B = modules.compl_div(ex, ey).repeat(1, 1, dudx1.shape[-1] // 2)
    C = modules.compl_mul(ex, ey).repeat(1, 1, dudx1.shape[-1] // 2)

    a, _ = diff_operators.jacobian(modules.compl_mul(A, dudx1), x)
    b, _ = diff_operators.jacobian(modules.compl_mul(B, dudx2), x)

    a = a[..., 0]
    b = b[..., 1]
    c = modules.compl_mul(modules.compl_mul(C, squared_slowness), wavenumber ** 2 * y)

    diff_constraint_hom = a + b + c
    diff_constraint_on = torch.where(source_boundary_values != 0.,
                                     diff_constraint_hom - source_boundary_values,
                                     torch.zeros_like(diff_constraint_hom))
    diff_constraint_off = torch.where(source_boundary_values == 0.,
                                      diff_constraint_hom,
                                      torch.zeros_like(diff_constraint_hom))
    if full_waveform_inversion:
        data_term = torch.where(rec_boundary_values != 0, y - rec_boundary_values, torch.Tensor([0.]).cuda())
    else:
        data_term = torch.Tensor([0.])

        if 'pretrain' in gt:  # we are not trying to solve for velocity
            data_term = pred_squared_slowness - squared_slowness[..., 0]

    return {'diff_constraint_on': torch.abs(diff_constraint_on).sum() * batch_size / 1e3,
            'diff_constraint_off': torch.abs(diff_constraint_off).sum(),
            'data_term': torch.abs(data_term).sum() * batch_size / 1}


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


def sdf_on_off_surf(model_output, gt):
    '''
    x: batch of input coordinates
    y: usually the output of the trial_soln function
    '''
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    gradient = diff_operators.gradient(pred_sdf, coords)
    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    sdf_constraint_on_surf = torch.where(gt_sdf == 0, pred_sdf, torch.zeros_like(pred_sdf))
    sdf_constraint_off_surf = torch.where(gt_sdf != 0, (gt_sdf - pred_sdf) ** 2, torch.zeros_like(pred_sdf))
#    sdf_constraint = (gt_sdf - pred_sdf)**2
    normal_constraint = torch.where(gt_sdf == 0, 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None], torch.zeros_like(gradient[..., :1]))
    grad_constraint = (gradient.norm(dim=-1) - 1.) ** 2

    # Exp      # Lapl
    # -----------------
    return {'sdf_on_surf': (sdf_constraint_on_surf ** 2).mean() * 3e3,
            'sdf_off_surf': sdf_constraint_off_surf.mean() * 1e2,
            'normal_constraint': normal_constraint.mean() * 1e1,
            'grad_constraint': grad_constraint.mean() * 1e1}


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


# inter = 3e3 for ReLU-PE

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
    curv_constraint = torch.where(abs_curvature> 10., curv_constraint, torch.zeros_like(pred_curvature))

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
