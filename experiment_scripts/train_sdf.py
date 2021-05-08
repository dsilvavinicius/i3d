'''Reproduces Sec. 4.2 in main paper and Sec. 4 in Supplement.
'''

# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, meta_modules, utils, training, loss_functions, modules

from torch.utils.data import DataLoader
import configargparse

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=1400)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=5e-5')
p.add_argument('--num_epochs', type=int, default=10000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=1,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')
p.add_argument('--point_cloud_path', type=str, default='/home/sitzmann/data/point_cloud.xyz',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--time', action='store_true', required=False, help='Indicates that time will also be considered')

opt = p.parse_args()

sdf_dataset = None
in_features = 0
loss_fn = None

if(opt.time == True):
    sdf_dataset = dataio.PointCloudImplictFunctions_4D(opt.point_cloud_path, on_surface_points=opt.batch_size) # for tubular vicinity
    in_features = 4
    loss_fn = loss_functions.implicit_function_4D
    summary_fn = None
else:
    #sdf_dataset = dataio.PointCloud(opt.point_cloud_path, on_surface_points=opt.batch_size)
    #sdf_dataset = dataio.PointCloudPrincipalDirections(opt.point_cloud_path, on_surface_points=opt.batch_size)
    #sdf_dataset = dataio.PointCloudTubularCurvatures(opt.point_cloud_path, on_surface_points=opt.batch_size)
    #sdf_dataset = dataio.PointCloudTubular(opt.point_cloud_path, on_surface_points=opt.batch_size) # for tubular vicinity
    sdf_dataset = dataio.PointCloudImplictFunctions(opt.point_cloud_path, on_surface_points=opt.batch_size) # for tubular vicinity
    in_features = 3
    #loss_fn = loss_functions.sdf_tensor_curvature
    #loss_fn = loss_functions.sdf_mean_curvature
    #loss_fn = loss_functions.sdf_gaussian_curvature
    #loss_fn = loss_functions.sdf_principal_curvatures
    #loss_fn = loss_functions.sdf_original
    #loss_fn = loss_functions.sdf_principal_directions
    #loss_fn = loss_functions.sdf_principal_curvature_segmentation
    #loss_fn = loss_functions.sdf_original_on_surface
    loss_fn = loss_functions.implicit_function
    summary_fn = utils.write_sdf_summary

dataloader = DataLoader(sdf_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)

# Define the model
if opt.model_type == 'nerf':
    model = modules.SingleBVPNet(type='relu', mode='nerf', in_features=in_features)
else:
    model = modules.SingleBVPNet(type=opt.model_type, in_features=in_features)
model.cuda()

root_path = os.path.join(opt.logging_root, opt.experiment_name)

training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, double_precision=False,
               clip_grad=True)