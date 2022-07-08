'''From the DeepSDF repository https://github.com/facebookresearch/DeepSDF
'''

import numpy as np
import plyfile
from skimage.measure import marching_cubes
import time
import torch


def gen_mc_coordinate_grid(N: int, voxel_size: float, t: float = None,
                           device: str = "cpu",
                           voxel_origin: list = [-1, -1, -1]) -> torch.Tensor:
    """Creates the coordinate grid for inference and marching cubes run.

    Parameters
    ----------
    N: int
        Number of elements in each dimension. Total grid size will be N ** 3

    voxel_size: number
        Size of each voxel

    t: float, optional
        Reconstruction time. Required for space-time models. Default value is
        None, meaning that time is not a model parameter

    device: string, optional
        Device to store tensors. Default is CPU

    voxel_origin: list[number, number, number], optional
        Origin coordinates of the volume. Must be the (bottom, left, down)
        coordinates. Default is [-1, -1, -1]

    Returns
    -------
    samples: torch.Tensor
        A (N**3, 3) shaped tensor with samples' coordinates. If t is not None,
        then the return tensor is has 4 columns instead of 3, with the last
        column equalling `t`.
    """
    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())

    sdf_coord = 3
    if t is not None:
        sdf_coord = 4

    # (x,y,z,sdf) if we are not considering time
    # (x,y,z,t,sdf) otherwise
    samples = torch.zeros(N ** 3, sdf_coord + 1, device=device,
                          requires_grad=False)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    # adding the time
    if t is not None:
        samples[:, sdf_coord-1] = t

    return samples


def create_mesh(
    decoder,
    filename="",
    t=-1, # time=-1 means we are only in the space
    N=256,
    max_batch=64 ** 3,
    offset=None,
    scale=None,
    device="cpu",
    silent=False
):
    decoder.eval()
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not
    # the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    samples = gen_mc_coordinate_grid(N, voxel_size, t=None if t == -1 else t,
                                     device=device)

    sdf_coord = 3
    if (t != -1):
        sdf_coord = 4

    num_samples = N ** 3
    head = 0

    start = time.time()
    while head < num_samples:
        # print(head)
        sample_subset = samples[head:min(head + max_batch, num_samples), 0:sdf_coord]

        samples[head:min(head + max_batch, num_samples), sdf_coord] = (
            decoder(sample_subset)["model_out"]
            .squeeze()
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, sdf_coord]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    if not silent:
        print(f"Sampling took: {end-start} s")

    verts, faces, normals, values = convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        offset,
        scale,
    )

    if filename:
        if not silent:
            print(f"Saving mesh to {filename}")

        save_ply(verts, faces, filename)

        if not silent:
            print("Done")

    return verts, faces, normals, values


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    if isinstance(pytorch_3d_sdf_tensor, torch.Tensor):
        numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()
    else:
        numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)

    # Check if the cubes contains the zero-level set
    level = 0.0
    if level < numpy_3d_sdf_tensor.min() or level > numpy_3d_sdf_tensor.max():
        print(f"Surface level must be within volume data range.")
    else:
        verts, faces, normals, values = marching_cubes(
            numpy_3d_sdf_tensor, level, spacing=[voxel_size] * 3
        )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    return mesh_points, faces, normals, values


def save_ply(verts, faces, filename):
    # try writing to the ply file
    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros(
        (num_verts,),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]
    )

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(verts[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(
        faces_building,
        dtype=[("vertex_indices", "i4", (3,))]
    )

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(filename)
