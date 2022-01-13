#!/usr/bin/env python
# coding: utf-8

import sys
import os
import json
import torch
from torch.utils.data import DataLoader
import configargparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataio
import utils
import training
import loss_functions
import modules


def parse_list_cmd_arg(cmdarg, sep=",", typ=int):
    parsed_arg = cmdarg
    if not isinstance(parsed_arg, list):
        parsed_arg = parsed_arg.split(sep)
    for i in range(len(parsed_arg)):
        parsed_arg[i] = typ(parsed_arg[i])
    return parsed_arg


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

p.add_argument('--w0', type=int, default=30,
               help='Multiplicative factor for the frequencies')
p.add_argument('--mesh_path', type=str, default='./data/armadillo.ply',
               help='Mesh input path')

p.add_argument('--xyz_path', type=str, default='./data/armadillo_tensor.xyz',
               help='Mesh input path')

p.add_argument("--percentiles", default=[70, 95],
    help="Curvature percentiles to split into low, medium and high curvatures.")

p.add_argument("--curvature_fractions", default=[0.6, 0.2, 0.2],
    help="Fraction of points with low, medium and high curvature per batch.")

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
opt = p.parse_args()

percentiles = parse_list_cmd_arg(opt.percentiles)
curvature_fracs = parse_list_cmd_arg(opt.curvature_fractions, typ=float)

available = torch.cuda.is_available()
print(f"CUDA available? {available}", flush=True)

device = torch.device("cuda:0" if available else "cpu")
print(f"Device: {device}")

# TODO: update dataio since we do not need the triangle mesh
sdf_dataset = dataio.PointCloudSDFPreComputedCurvaturesDirections(
    opt.mesh_path,
    opt.xyz_path,
    batch_size=opt.batch_size,
    scaling="bbox",
    uniform_sampling = True,
    low_med_percentiles=percentiles,
    curvature_fracs=curvature_fracs
)

dataloader = DataLoader(
    sdf_dataset,
    shuffle=True,
    batch_size=1,
    pin_memory=True,
    num_workers=0,
)

# launch the trained model
class SDFTrained(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define the model.
        self.model = modules.SingleBVPNet(
            typ="sine",
            final_layer_factor=1,
            in_features=3,
            hidden_features = 64,
            num_hidden_layers=1,
            w0=30
        )
        self.model.load_state_dict(torch.load('./logs/armadillo_b10000_w0-30_rede-1x64/checkpoints/model_final.pth'))
        self.model.cuda()

    def forward(self, coords):
        model_in = {'coords': coords}
        return self.model(model_in)#['model_out']

trained_model = SDFTrained()

# Define the local model.
model = modules.SingleBVPNet(typ="sine", hidden_features=256,
                              num_hidden_layers=2, in_features=3, w0=opt.w0)
model.to(device)

print(f"Is model on GPU? {next(model.parameters()).is_cuda}", flush=True)


# Define the loss
loss_fn = loss_functions.loss_mean_curvature(trained_model)
summary_fn = utils.write_sdf_summary

root_path = os.path.join(opt.logging_root, opt.experiment_name)

training.train(
    model=model,
    train_dataloader=dataloader,
    epochs=opt.num_epochs,
    lr=opt.lr,
    steps_til_summary=opt.steps_til_summary,
    epochs_til_checkpoint=opt.epochs_til_ckpt,
    model_dir=root_path,
    loss_fn=loss_fn,
    summary_fn=summary_fn,
    double_precision=False,
    clip_grad=True,
    device=device
)
