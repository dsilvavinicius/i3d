'''Test script for experiments in paper Sec. 4.2, Supplement Sec. 3, reconstruction from laplacian.
'''

# Enable import from parent package
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import modules, utils
import sdf_meshing
import configargparse
import diff_operators
import implicit_functions

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=16384)
p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')

p.add_argument('--resolution', type=int, default=256)
p.add_argument('--time', action='store_true', required=False, help='Indicates that time will also be considered')

p.add_argument('--w0', type=int, default=30,
               help='Multiplicative factor for the frequencies')

opt = p.parse_args()

in_features = 3


class SDFDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define the model.
        self.model = modules.SingleBVPNet(
            typ="sine",
            final_layer_factor=1,
            in_features=in_features,
            num_hidden_layers=3,
            w0=opt.w0
        )
        self.model.load_state_dict(torch.load(opt.checkpoint_path))
        self.model.cuda()

    def forward(self, coords):
        model_in = {'coords': coords}
        return self.model(model_in)#['model_out']


sdf_decoder = SDFDecoder()
#sdf_decoder = implicit_functions.torus()
#sdf_decoder = implicit_functions.double_torus()

root_path = os.path.join(opt.logging_root, opt.experiment_name)
utils.cond_mkdir(root_path)

sdf_meshing.create_mesh_with_curvatures(sdf_decoder, os.path.join(root_path, 'test'), N=opt.resolution)
