# coding: utf-8

import torch
import torch.nn.functional as F
import diff_operators


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


def off_surface_without_sdf_constraint(gt_sdf, pred_sdf, radius=1e2):
    """
    This function penalizes the pred_sdf of points in gt_sdf!=0
    Used in SIREN's paper
    """
    return torch.where(
           gt_sdf == 0,
           torch.zeros_like(pred_sdf),
           torch.exp(-radius * torch.abs(pred_sdf))
        )


def on_surface_normal_constraint(gt_sdf, gt_normals, grad):
    """
    This function return a number that measure how far gt_normals
    and grad are aligned in the zero-level set of sdf.
    """
    return torch.where(
           gt_sdf == 0,
           1 - F.cosine_similarity(grad, gt_normals, dim=-1)[..., None],
           torch.zeros_like(grad[..., :1])
    )


def sdf_sitzmann(X, gt):
    """Loss function employed in Sitzmann et al. for SDF experiments [1].

    Parameters
    ----------
    X: dict[str=>torch.Tensor]
        Model output with the following keys: 'model_in' and 'model_out'
        with the model input and SDF values respectively.

    gt: dict[str=>torch.Tensor]
        Ground-truth data with the following keys: 'sdf' and 'normals', with
        the actual SDF values and the input data normals, respectively.

    Returns
    -------
    loss: dict[str=>torch.Tensor]
        The calculated loss values for each constraint.

    References
    ----------
    [1] Sitzmann, V., Martel, J. N. P., Bergman, A. W., Lindell, D. B.,
    & Wetzstein, G. (2020). Implicit Neural Representations with Periodic
    Activation Functions. ArXiv. Retrieved from http://arxiv.org/abs/2006.09661
    """
    gt_sdf = gt["sdf"]
    gt_normals = gt["normals"]

    # print("GT_SDF SIZE = ", gt_sdf.size())
    # print("GT_NORMALS SIZE = ", gt_normals.size())

    coords = X["model_in"]
    pred_sdf = X["model_out"]

    # print("coords SIZE = ", coords.size())
    # print("pred_sdf SIZE = ", pred_sdf.size())

    grad = diff_operators.gradient(pred_sdf, coords)

    # Initial-boundary constraints
    sdf_constraint = sdf_constraint_on_surf(gt_sdf, pred_sdf)
    inter_constraint = off_surface_without_sdf_constraint(gt_sdf, pred_sdf)
    normal_constraint = on_surface_normal_constraint(gt_sdf, gt_normals, grad)

    # PDE constraints
    grad_constraint = eikonal_constraint(grad)

    return {
        "sdf_constraint": sdf_constraint.mean() * 3e3,
        "inter_constraint": inter_constraint.mean() * 1e2,
        "normal_constraint": normal_constraint.mean() * 1e2,
        "grad_constraint": grad_constraint.mean() * 5e1,
    }


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
    return {
       'sdf_on_surf': sdf_constraint_on_surf(gt_sdf, pred_sdf).mean() * 3e3,
       'sdf_off_surf': sdf_constraint_off_surf(gt_sdf, pred_sdf).mean() * 2e2,
       'normal_constraint': vector_aligment_on_surf(gt_sdf, gt_normals, gradient).mean() *1e2 ,#* 1e1,
       'grad_constraint': eikonal_constraint(gradient).mean() * 5e1#1e1
    }


# learn a second model that add details to a previously trained model
class loss_add_detail(torch.nn.Module):
    def __init__(self, trained_model):
        super().__init__()
        # Define the model.
        self.model = trained_model
        self.model.cuda()

    def forward(self, model_output, gt):
        gt_sdf = gt['sdf']
        gt_normals = gt['normals']

        #local
        coords = model_output['model_in']
        pred_sdf = model_output['model_out']

        trained_model = self.model(coords)
        trained_coords = trained_model['model_in']
        trained_sdf = trained_model['model_out']

        gradient = diff_operators.gradient(pred_sdf, coords) + diff_operators.gradient(trained_sdf, trained_coords)

        # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
        return {'sdf_on_surf': sdf_constraint_on_surf(gt_sdf, pred_sdf + trained_sdf).mean() * 3e3,
                'sdf_off_surf': sdf_constraint_off_surf(gt_sdf, pred_sdf + trained_sdf).mean() * 2e2,
                'normal_constraint': vector_aligment_on_surf(gt_sdf, gt_normals, gradient).mean() *1e2 ,#* 1e1,
                'grad_constraint': eikonal_constraint(gradient).mean() * 5e1}#1e1}


#learn the mean curvature of a neural surface
class loss_mean_curvature(torch.nn.Module):
    def __init__(self, trained_model):
        super().__init__()
        # Define the model.
        self.model = trained_model
        self.model.cuda()

    def forward(self, model_output, gt):
        coords = model_output['model_in']
        pred_curvature = model_output['model_out']

        trained_model = self.model(coords)
        global_coords = trained_model['model_in']
        global_sdf    = trained_model['model_out']

        # ground truth mean curvature
        curvature = diff_operators.mean_curvature(global_sdf, global_coords)
        constraint = (curvature - pred_curvature)**2

        return {'constraint': constraint.mean()}


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
    #dirs_constraint = torch.where(pred_dirs[0][...,3].unsqueeze(-1) == 0, dirs_constraint, torch.zeros_like(dirs_constraint))

    #removing problematic curvatures and planar points
    planar_curvature = 0.5*torch.abs(gt_min_curvature-gt_max_curvature)
    dirs_constraint = torch.where(planar_curvature > 10  , dirs_constraint, torch.zeros_like(dirs_constraint))
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
