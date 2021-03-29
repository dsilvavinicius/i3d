#!/usr/bin/env python

import sys
import os
import numpy as np
from torch.utils.data import DataLoader, BatchSampler
from configargparse import ArgumentParser
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataio
import meta_modules
import utils
import training
import loss_functions
import modules
from samplers import CurvatureSeqSampler


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
p.add_argument(
    "--sampler_type",
    type=str,
    default="biased-seq",
    help="Options are \"biased-seq\", \"biased-histogram\" and \"random\""
)

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
opt = p.parse_args()

sdf_dataset = dataio.PointCloudNonRandom(opt.point_cloud_path)

if opt.sampler_type == "biased-seq":
    sampler = CurvatureSeqSampler(sdf_dataset)
elif opt.sampler_type == "biased-histogram":
    raise NotImplementedError
else:
    raise NotImplementedError

if opt.batch_size > 1:
    dataloader = DataLoader(
        sdf_dataset,
        batch_sampler=BatchSampler(sampler, batch_size=opt.batch_size, drop_last=False),
        pin_memory=True,
        num_workers=0,
    )
else:
    dataloader = DataLoader(
        sdf_dataset,
        sampler=sampler,
        pin_memory=True,
        num_workers=0,
    )

# Define the model.
if opt.model_type == 'nerf':
    model = modules.SingleBVPNet(type='relu', mode='nerf', in_features=3)
else:
    model = modules.SingleBVPNet(type=opt.model_type, in_features=3)
model.cuda()

# Define the loss
loss_fn = loss_functions.sdf_original
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
