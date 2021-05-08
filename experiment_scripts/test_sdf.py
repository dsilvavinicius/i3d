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

p.add_argument('--model_type', type=str, default='sine',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')
p.add_argument('--mode', type=str, default='mlp',
               help='Options are "mlp" or "nerf"')
p.add_argument('--resolution', type=int, default=1600)
p.add_argument('--time', action='store_true', required=False, help='Indicates that time will also be considered')

opt = p.parse_args()


in_features = 0

if(opt.time == True):
    in_features = 4
else:
    in_features = 3

class SDFDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define the model.
        if opt.mode == 'mlp':
            self.model = modules.SingleBVPNet(type=opt.model_type, final_layer_factor=1, in_features=in_features)
        elif opt.mode == 'nerf':
            self.model = modules.SingleBVPNet(type='relu', mode='nerf', final_layer_factor=1, in_features=in_features)
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

if(opt.time == True):
    N = 5 #number of samples of the interval time [0,1]
    for i in range(N):
        t = i/(N-1)
        sdf_meshing.create_mesh(sdf_decoder, os.path.join(root_path, f"test_epoch_{t}"), t, N=opt.resolution)
else:
    sdf_meshing.create_mesh_with_curvatures(sdf_decoder, os.path.join(root_path, 'test'), N=opt.resolution)
