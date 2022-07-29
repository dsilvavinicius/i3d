# import plyfile
import open3d as o3d
# import open3d.core as o3c
import numpy as np
import torch
import diff_operators
from model import SIREN
from util import siren_v1_to_v2
from meshing import save_ply


mesh_map = {
    "armadillo": "./data/armadillo.ply",
    "bunny": "./data/bunny.ply",
    "dragon": "./data/dragon.ply",
    "happy_buddha": "./data/happy_buddha.ply",
    "lucy": "./data/lucy_simple.ply",
}

MESH_TYPE = "armadillo"

mesh = o3d.io.read_triangle_mesh(mesh_map[MESH_TYPE])
mesh.compute_vertex_normals()
mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
print(mesh)

coords = torch.from_numpy(mesh.vertex["positions"].numpy())

decoder = SIREN(3, 1, [256, 256, 256, 256], w0=30)

weights_file = "./results/armadillo_biased_curvature_sdf/models/model_best.pth"
weights = torch.load(weights_file, map_location=torch.device("cuda:0"))
try:
    decoder.load_state_dict(weights)
except RuntimeError:
    new_weights, diff = siren_v1_to_v2(weights, True)
    print(diff)
    new_weights_file = weights_file.split(".")[0] + "_v2.pth"
    torch.save(new_weights, weights_file)
    decoder.load_state_dict(new_weights)

decoder.eval()

model = decoder(coords)
X = model['model_in']
y = model['model_out']

curvatures = diff_operators.mean_curvature(y, X)
verts = np.hstack((coords.detach().numpy(), mesh.vertex["normals"].numpy(), curvatures.detach().numpy()))
print(verts.shape)

faces = mesh.triangle["indices"].numpy()

save_ply(verts, faces, "./data/armadillo_test_curvs.ply",
         attrs=[("nx", "f4"), ("ny", "f4"), ("nz", "f4"), ("quality", "f4")])
