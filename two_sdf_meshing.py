'''From the DeepSDF repository https://github.com/facebookresearch/DeepSDF
'''
#!/usr/bin/env python3

import logging
from pickle import FALSE
import numpy as np
from numpy.core.fromnumeric import shape, squeeze
import plyfile
import skimage.measure
import time
import torch
from torch._C import device

import diff_operators
import modules


def create_mesh(
    decoder1, decoder2, filename, 
    add_detail = True, add_vertex_function = False, 
    N=256, max_batch=64 ** 3, offset=None, scale=None
):
    start = time.time()
    ply_filename = filename

    decoder1.eval()
    decoder2.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

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

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        print(head)
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()

        #model_in = {'coords': sample_subset}
        model_in = sample_subset
        
        if add_detail:
            #the details of decoder1 are in decoder2
            decoder_sample_subset = decoder1(model_in)['model_out'] + decoder2(model_in)['model_out']
        else:
            decoder_sample_subset = decoder1(model_in)['model_out']

        samples[head : min(head + max_batch, num_samples), 3] = (
            decoder_sample_subset
            .squeeze()#.squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    print("sampling takes: %f" % (end - start))

    convert_sdf_samples_to_ply(
    #convert_sdf_samples_to_ply_with_curvatures_directions(
        decoder1,
        decoder2,
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
        add_vertex_function,
        offset,
        scale,
    )

def compute_mesh_curvatures(decoder, mesh_points):
    num_verts = mesh_points.shape[0]
    coords = torch.from_numpy(mesh_points).float().cuda()
    pred_curvature = []
    N = 200
    for i in range(N):
        coords_i = coords[int(num_verts*i/N): int(num_verts*(i+1)/N),:]
        #model_output_i = decoder({'coords': coords_i.unsqueeze(0)}) #to use during training
        model_output_i = decoder(coords_i.unsqueeze(0))

        coords_i = model_output_i['model_in']
        sdf_vert_values_i = model_output_i['model_out']
        
        #gradient = diff_operators.gradient(sdf_vert_values_i, coords_i)
        #hessian = diff_operators.hessian(sdf_vert_values_i, coords_i)

        # principal curvatures
        #pred_curvatures_i = diff_operators.principal_curvature(sdf_vert_values_i, coords_i, gradient, hessian)
        #pred_curvature_i = pred_curvatures_i[1].cpu().detach().numpy()
        
        #pred_curvature_i = diff_operators.principal_curvature_region_detection(sdf_vert_values_i, coords_i).cpu().detach().numpy()
        #pred_curvature_i = diff_operators.umbilical_indicator(sdf_vert_values_i, coords_i).cpu().detach().numpy()
        #pred_curvature_i = diff_operators.gaussian_curvature(gradient, hessian).unsqueeze(-1).cpu().detach().numpy()
        pred_curvature_i = diff_operators.mean_curvature(sdf_vert_values_i, coords_i).squeeze(0).cpu().detach().numpy()
        if len(pred_curvature)==0:
            pred_curvature = pred_curvature_i
        else:
            pred_curvature = np.concatenate((pred_curvature, pred_curvature_i), axis=0)
    
    return pred_curvature

def compute_mesh_curvature_directions(decoder, mesh_points):
    num_verts = mesh_points.shape[0]
    coords = torch.from_numpy(mesh_points).float().cuda()
    direction = []
    N = 200
    for i in range(N):
        coords_i = coords[int(num_verts*i/N): int(num_verts*(i+1)/N),:]
        #model_output_i = decoder({'coords': coords_i.unsqueeze(0)}) #to use during training
        model_output_i = decoder(coords_i.unsqueeze(0))

        coords_i = model_output_i['model_in']
        sdf_vert_values_i = model_output_i['model_out']
    
        gradient = diff_operators.gradient(sdf_vert_values_i, coords_i)
        hessian = diff_operators.hessian(sdf_vert_values_i, coords_i)

        # principal directions
        #curvature_i = diff_operators.principal_curvature(sdf_vert_values_i, coords_i, gradient, hessian)[0]
        direction_i = diff_operators.principal_directions(gradient, hessian[0])[0][...,0:3]       
        #direction_i = torch.where(curvature_i < -0.5, direction_i, torch.zeros_like(direction_i))
        #direction_i = gradient

        direction_i = direction_i.squeeze(0).cpu().detach().numpy()
        
        if len(direction)==0:
            direction = direction_i
        else:
            direction = np.concatenate((direction, direction_i), axis=0)
    
    return direction


def convert_sdf_samples_to_ply(
    decoder1,
    decoder2,
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    add_vertex_function=False,
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

    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
            numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
        )
    except:
        pass

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

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("quality", "f4")])
    
    if add_vertex_function:
        # considers the function encoded in decoder2
        coords = torch.from_numpy(mesh_points).float().cuda()
        pred_curvatures = decoder2(coords.unsqueeze(0))['model_out'].squeeze(0).cpu().detach().numpy()#compute_mesh_curvatures(decoder, mesh_points)
    else:
        # considers the curvatures of mesh_points
        pred_curvatures = compute_mesh_curvatures(decoder1, mesh_points)
    
    for i in range(0, num_verts):
        vertex = np.array([mesh_points[i, 0],mesh_points[i, 1], mesh_points[i, 2],pred_curvatures[i]]) 
        verts_tuple[i] = tuple(vertex)

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )


def convert_sdf_samples_to_ply_with_curvatures_directions(
    decoder,
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
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

    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
            numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
        )
    except:
        pass

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

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),])

    # computing the principal directions of mesh_points ----------------
    pred_curvatures = compute_mesh_curvature_directions(decoder, mesh_points)
    #---------------------------------------------------------

    for i in range(0, num_verts):
        vertex = np.array([mesh_points[i, 0],mesh_points[i, 1], mesh_points[i, 2],pred_curvatures[i, 0],pred_curvatures[i, 1],pred_curvatures[i, 2]]) 
        verts_tuple[i] = tuple(vertex)

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )