#!/usr/bin/env python

import sys
import os
import numpy as np
from torch.utils.data import DataLoader, Sampler
from configargparse import ArgumentParser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataio
import meta_modules
import utils
import training
import loss_functions
import modules


class CurvatureSeqSampler(Sampler):
    """A simple sampler that orders the data samples by curvature and returns
    them in sequence.

    Parameters
    ----------
    data_source: PointCloud
        The actual data source. Must have the curvatures as features.
    """
    def __init__(self, data_source):
        self.data_source = data_source
        self.sorted_curvatures_idx = np.argsort(np.absolute(self.data_source.curvatures))

    def __iter__(self):
        return iter(range(len(self.sorted_curvatures_idx)))

    def __len__(self):
        return len(self.data_source)


p = ArgumentParser()
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
opt = p.parse_args()

sdf_dataset = dataio.PointCloud(
    opt.point_cloud_path,
    on_surface_points=opt.batch_size
)

curv_sampler = CurvatureSeqSampler(sdf_dataset)
dataloader = DataLoader(
    sdf_dataset,
    sampler=curv_sampler,
    pin_memory=True,
    num_workers=0
)

# Define the model.
if opt.model_type == 'nerf':
    model = modules.SingleBVPNet(type='relu', mode='nerf', in_features=3)
else:
    model = modules.SingleBVPNet(type=opt.model_type, in_features=3)
model.cuda()

# Define the loss
loss_fn = loss_functions.sdf
summary_fn = utils.write_sdf_summary

root_path = os.path.join(opt.logging_root, opt.experiment_name)

training.train(model=model,
               train_dataloader=dataloader,
               epochs=opt.num_epochs,
               lr=opt.lr,
               steps_til_summary=opt.steps_til_summary,
               epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path,
               loss_fn=loss_fn,
               summary_fn=summary_fn,
               double_precision=False,
               clip_grad=True)
