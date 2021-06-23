# coding: utf-8

import numpy as np
import torch
import diff_operators
import implicit_functions
from torch.utils.data import Dataset


class PointCloud(Dataset):
    def __init__(self, pointcloud_path, on_surface_points, keep_aspect_ratio=True):
        super().__init__()

        print("Loading point cloud")
        point_cloud = np.genfromtxt(pointcloud_path)

        k_range = 5000
        n = point_cloud.shape[0]
        point_cloud = point_cloud[np.absolute(point_cloud[:, 3]) <= k_range]
        out_min_k1 = n - point_cloud.shape[0]

        point_cloud = point_cloud[np.absolute(point_cloud[:, 4]) <= k_range]
        out_min_k2 = n - point_cloud.shape[0] + out_min_k1

        print(f"[WARN] Removed {out_min_k1} with abs(k1) > {k_range}.")
        print(f"[WARN] Removed {out_min_k2} with abs(k2) > {k_range}.")

        print("Finished loading point cloud")

        #exporting ply (point, curvature, normal):  x, y, z, k, nx, ny, nz
        #coords = point_cloud[:, :3]
        #curvatures = point_cloud[:, 3]
        #self.normals = point_cloud[:, 4:7]

        #exporting ply (point, curvatures, normal):  x, y, z, k1, k2, nx, ny, nz
        point_cloud = point_cloud[np.absolute(point_cloud[:, 3]) < 10000]
        point_cloud = point_cloud[np.absolute(point_cloud[:, 4]) < 10000]

        coords = point_cloud[:, :3]
        min_curvatures = point_cloud[:, 4]
        max_curvatures = point_cloud[:, 3]
        self.normals = point_cloud[:, 5:8]

        #for mesh lab curvatures
        #curvatures = point_cloud[:, 6]
        #self.normals = point_cloud[:, 3:6]

        # Reshape point cloud such that it lies in bounding box of (-1, 1) (distorts geometry, but makes for high
        # sample efficiency)
        coords -= np.mean(coords, axis=0, keepdims=True)
        if keep_aspect_ratio:
            coord_max = np.amax(coords)
            coord_min = np.amin(coords)
        else:
            coord_max = np.amax(coords, axis=0, keepdims=True)
            coord_min = np.amin(coords, axis=0, keepdims=True)

        self.coords = (coords - coord_min) / (coord_max - coord_min)
        self.coords -= 0.5
        self.coords *= 2.

        self.min_curvatures = min_curvatures
        self.max_curvatures = max_curvatures

        self.on_surface_points = on_surface_points

    def __len__(self):
        return self.coords.shape[0] // self.on_surface_points

    def __getitem__(self, idx):
        point_cloud_size = self.coords.shape[0]

        off_surface_samples = self.on_surface_points  # **2
        total_samples = self.on_surface_points + off_surface_samples

        # Random coords
        rand_idcs = np.random.choice(point_cloud_size, size=self.on_surface_points)

        on_surface_coords = self.coords[rand_idcs, :]
        on_surface_normals = self.normals[rand_idcs, :]
        on_surface_min_curvature = self.min_curvatures[rand_idcs]
        on_surface_max_curvature = self.max_curvatures[rand_idcs]

        off_surface_coords = np.random.uniform(-1, 1, size=(off_surface_samples, 3))
        off_surface_normals = np.ones((off_surface_samples, 3)) * -1
        off_surface_min_curvature = np.zeros((off_surface_samples))
        off_surface_max_curvature = np.zeros((off_surface_samples))
        # We consider the curvature of the sphere centered in the origin with radius equal to the norm of the coordinate.
        # off_surface_curvature = 1 / (np.linalg.norm(off_surface_coords, axis=1) ** 2)

        sdf = np.zeros((total_samples, 1))  # on-surface = 0
        sdf[self.on_surface_points:, :] = -1  # off-surface = -1

        coords = np.concatenate((on_surface_coords, off_surface_coords), axis=0)
        normals = np.concatenate((on_surface_normals, off_surface_normals), axis=0)
        min_curvature = np.concatenate((on_surface_min_curvature, off_surface_min_curvature))
        min_curvature = np.expand_dims(min_curvature, -1)
        max_curvature = np.concatenate((on_surface_max_curvature, off_surface_max_curvature))
        max_curvature = np.expand_dims(max_curvature, -1)


        return {'coords': torch.from_numpy(coords).float()}, {'sdf': torch.from_numpy(sdf).float(),
                                                              'normals': torch.from_numpy(normals).float(),
                                                              'min_curvature': torch.from_numpy(min_curvature).float(),
                                                              'max_curvature': torch.from_numpy(max_curvature).float()}


class PointCloudTubular(Dataset):
    def __init__(self, pointcloud_path, on_surface_points, keep_aspect_ratio=True):
        super().__init__()

        print("Loading point cloud")
        point_cloud = np.genfromtxt(pointcloud_path)
        print("Finished loading point cloud")

        #exporting ply (point, curvatures, normal):  x, y, z, k1, k2, nx, ny, nz
        coords = point_cloud[:, :3]
        #min_curvatures = point_cloud[:, 4]
        #max_curvatures = point_cloud[:, 3]
        self.normals = point_cloud[:, 5:8]

        # Reshape point cloud such that it lies in bounding box of (-1, 1) (distorts geometry, but makes for high
        # sample efficiency)
        coords -= np.mean(coords, axis=0, keepdims=True)
        if keep_aspect_ratio:
            coord_max = np.amax(coords)
            coord_min = np.amin(coords)
        else:
            coord_max = np.amax(coords, axis=0, keepdims=True)
            coord_min = np.amin(coords, axis=0, keepdims=True)

        self.coords = (coords - coord_min) / (coord_max - coord_min)
        self.coords -= 0.5
        self.coords *= 2.

        #self.min_curvatures = min_curvatures
        #self.max_curvatures = max_curvatures

        self.on_surface_points = on_surface_points

    def __len__(self):
        return self.coords.shape[0] // self.on_surface_points

    def __getitem__(self, idx):
        point_cloud_size = self.coords.shape[0]

        off_surface_samples = self.on_surface_points 
        in_surface_samples  = self.on_surface_points  
        out_surface_samples = self.on_surface_points 
        total_samples = in_surface_samples + self.on_surface_points + out_surface_samples + off_surface_samples
      
        # Random coords
        rand_idcs = np.random.choice(point_cloud_size, size=self.on_surface_points)

        on_surface_coords = self.coords[rand_idcs, :]
        on_surface_normals = self.normals[rand_idcs, :]

        #tubular vicinity
        epsilon = 0.0001
        in_surface_coords  = on_surface_coords - epsilon*on_surface_normals
        out_surface_coords = on_surface_coords + epsilon*on_surface_normals

        off_surface_coords = np.random.uniform(-1, 1, size=(off_surface_samples, 3))
        off_surface_normals = np.ones((off_surface_samples, 3)) * -1
        # We consider the curvature of the sphere centered in the origin with radius equal to the norm of the coordinate.
        # off_surface_curvature = 1 / (np.linalg.norm(off_surface_coords, axis=1) ** 2)

        sdf = np.zeros((total_samples, 1))#on-surface = 0 
        sdf[in_surface_samples + self.on_surface_points + out_surface_samples:, :] = -1  # off-surface = -1
        sdf[in_surface_samples + self.on_surface_points : in_surface_samples +  self.on_surface_points + out_surface_samples , :] = epsilon  # out-surface = epsilon
        sdf[ : in_surface_samples , :] = -epsilon  # in-surface = -epsilon

        # coordinates of the neighborhood of the tubular vicinity + off_surface
        coords = np.concatenate((in_surface_coords, on_surface_coords), axis=0)
        coords = np.concatenate((coords, out_surface_coords), axis=0)
        coords = np.concatenate((coords, off_surface_coords), axis=0)

        # duplicate the normals
        normals = np.concatenate((on_surface_normals, on_surface_normals), axis=0)
        normals = np.concatenate((normals, on_surface_normals), axis=0)
        normals = np.concatenate((normals, off_surface_normals), axis=0)

        return {'coords': torch.from_numpy(coords).float()}, {'sdf': torch.from_numpy(sdf).float(),
                                                              'normals': torch.from_numpy(normals).float()}


class PointCloudNonRandom(Dataset):
    """Point Cloud dataset where the sampling is done by a proper sampler
    instead of inside __getitem__. The goal is to decouple the dataset
    from the sampling strategy, allowing us to experiment with diferent
    strategies.

    Parameters
    ----------
    pointcloud_path: str
        Path to the input file. This file is loaded as a numpy array. We assume
        that the data is organized as follows: x, y, z, nx, ny, nz, k1, k2, sdf

    keep_aspect_ratio: boolean, optional
        Indicates whether the mesh aspect ratio will be mantained when
        reshaping it to fit in a bounding box of size 2 (-1, 2). Default is
        True.

    k_range: int, optional
        The maximum curvature value to allow. Any samples with absolute value
        of curvature larger than this will be REMOVED from the cloud. Default
        value is 10000.

    See Also
    --------
    numpy.genfromtxt
    """
    def __init__(self, pointcloud_path, keep_aspect_ratio=True, k_range=10000):
        super().__init__()

        print("Loading point cloud")
        point_cloud = np.genfromtxt(pointcloud_path)
        print("Finished loading point cloud")

        # exporting ply (point, curvatures, normal):  x, y, z, k1, k2, nx, ny, nz
        # Removing points with absurd curvatures.
        n = point_cloud.shape[0]
        point_cloud = point_cloud[np.absolute(point_cloud[:, 3]) <= k_range]
        out_min_k1 = n - point_cloud.shape[0]

        point_cloud = point_cloud[np.absolute(point_cloud[:, 4]) <= k_range]
        out_min_k2 = n - point_cloud.shape[0] + out_min_k1

        print(f"[WARN] Removed {out_min_k1} with abs(k1) > {k_range}.")
        print(f"[WARN] Removed {out_min_k2} with abs(k2) > {k_range}.")

        coords = point_cloud[:, :3]
        max_curvatures = point_cloud[:, 3]
        min_curvatures = point_cloud[:, 4]
        self.normals = point_cloud[:, 5:]

        # Reshape point cloud such that it lies in bounding box of (-1, 1) (distorts geometry, but makes for high
        # sample efficiency)
        coords -= np.mean(coords, axis=0, keepdims=True)
        if keep_aspect_ratio:
            coord_max = np.amax(coords)
            coord_min = np.amin(coords)
        else:
            coord_max = np.amax(coords, axis=0, keepdims=True)
            coord_min = np.amin(coords, axis=0, keepdims=True)

        self.coords = (coords - coord_min) / (coord_max - coord_min)
        self.coords -= 0.5
        self.coords *= 2.

        self.min_curvatures = min_curvatures
        self.max_curvatures = max_curvatures

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, idx):
        if not isinstance(idx, list):
            idx = [idx]

        on_surface_coords = self.coords[idx, :]
        on_surface_normals = self.normals[idx, :]
        on_surface_min_curvature = self.min_curvatures[idx]
        on_surface_max_curvature = self.max_curvatures[idx]

        off_surface_coords = np.random.uniform(-1, 1, size=(len(idx), 3))
        off_surface_normals = np.ones((len(idx), 3)) * -1
        off_surface_min_curvature = np.zeros((len(idx)))
        off_surface_max_curvature = np.zeros((len(idx)))

        sdf = np.zeros((2 * len(idx), 1))  # on-surface = 0
        sdf[len(idx):, :] = -1  # off-surface = -1

        coords = np.vstack((on_surface_coords, off_surface_coords))
        normals = np.vstack((on_surface_normals, off_surface_normals))
        min_curvature = np.concatenate((on_surface_min_curvature, off_surface_min_curvature))
        min_curvature = np.expand_dims(min_curvature, -1)
        max_curvature = np.concatenate((on_surface_max_curvature, off_surface_max_curvature))
        max_curvature = np.expand_dims(max_curvature, -1)

        return {
            "coords": torch.from_numpy(coords).float()
        }, {
            "sdf": torch.from_numpy(sdf).float(),
            "normals": torch.from_numpy(normals).float(),
            "min_curvature": torch.from_numpy(min_curvature),
            "max_curvature": torch.from_numpy(max_curvature)
        }


class PointCloudTubularCurvatures(Dataset):
    def __init__(self, pointcloud_path, on_surface_points, keep_aspect_ratio=True):
        super().__init__()

        print("Loading point cloud")
        point_cloud = np.genfromtxt(pointcloud_path)

        k_range = 5000

        n = point_cloud.shape[0]
        point_cloud = point_cloud[np.absolute(point_cloud[:, 3]) <= k_range]
        out_min_k1 = n - point_cloud.shape[0]

        point_cloud = point_cloud[np.absolute(point_cloud[:, 4]) <= k_range]
        out_min_k2 = n - point_cloud.shape[0] + out_min_k1

        print(f"[WARN] Removed {out_min_k1} with abs(k1) > {k_range}.")
        print(f"[WARN] Removed {out_min_k2} with abs(k2) > {k_range}.")

        print("Finished loading point cloud")

        #exporting ply (point, curvatures, normal):  x, y, z, k1, k2, nx, ny, nz
        coords = point_cloud[:, :3]
        min_curvatures = point_cloud[:, 4]
        max_curvatures = point_cloud[:, 3]
        self.normals = point_cloud[:, 5:8]

        # Reshape point cloud such that it lies in bounding box of (-1, 1) (distorts geometry, but makes for high
        # sample efficiency)
        coords -= np.mean(coords, axis=0, keepdims=True)
        if keep_aspect_ratio:
            coord_max = np.amax(coords)
            coord_min = np.amin(coords)
        else:
            coord_max = np.amax(coords, axis=0, keepdims=True)
            coord_min = np.amin(coords, axis=0, keepdims=True)

        self.coords = (coords - coord_min) / (coord_max - coord_min)
        self.coords -= 0.5
        self.coords *= 2.

        self.min_curvatures = min_curvatures
        self.max_curvatures = max_curvatures

    def __len__(self):
        return self.coords.shape[0] // self.on_surface_points

    def __getitem__(self, idx):
        point_cloud_size = self.coords.shape[0]

        off_surface_samples = self.on_surface_points
        in_surface_samples = self.on_surface_points
        out_surface_samples = self.on_surface_points
        total_samples = in_surface_samples + self.on_surface_points + out_surface_samples + off_surface_samples

        # Random coords
        rand_idcs = np.random.choice(point_cloud_size, size=self.on_surface_points)

        on_surface_coords = self.coords[rand_idcs, :]
        on_surface_normals = self.normals[rand_idcs, :]
        on_surface_min_curvature = np.expand_dims(self.min_curvatures[rand_idcs],-1)
        on_surface_max_curvature = np.expand_dims(self.max_curvatures[rand_idcs],-1)

        #tubular vicinity using curvature radius
        epsilon = 0.0005
        curvature_radius = 1./(np.maximum(np.absolute(on_surface_min_curvature), np.absolute(on_surface_max_curvature))) 
        curvature_radio = np.min(curvature_radius)
        curvature_radio = np.minimum(curvature_radio, epsilon)
       
        in_surface_coords  = on_surface_coords - curvature_radio*on_surface_normals
        out_surface_coords = on_surface_coords + curvature_radio*on_surface_normals

        in_surface_min_curvature, in_surface_max_curvature = diff_operators.principal_curvature_parallel_surface(on_surface_min_curvature,
                                                                                                                 on_surface_max_curvature, -curvature_radio)
        out_surface_min_curvature, out_surface_max_curvature = diff_operators.principal_curvature_parallel_surface(on_surface_min_curvature,
                                                                                                                   on_surface_max_curvature, curvature_radio)

        off_surface_coords = np.random.uniform(-1, 1, size=(off_surface_samples, 3))
        off_surface_normals = np.ones((off_surface_samples, 3)) * -1
        off_surface_min_curvature = np.zeros_like(on_surface_min_curvature)
        off_surface_max_curvature = np.zeros_like(on_surface_max_curvature)
        # We consider the curvature of the sphere centered in the origin with radius equal to the norm of the coordinate.
        # off_surface_curvature = 1 / (np.linalg.norm(off_surface_coords, axis=1) ** 2)

        sdf = np.zeros((total_samples, 1))#on-surface = 0 
        sdf[in_surface_samples + self.on_surface_points + out_surface_samples:, :] = -1  # off-surface = -1
        sdf[in_surface_samples + self.on_surface_points : in_surface_samples +  self.on_surface_points + out_surface_samples , :] = curvature_radio  # out-surface = epsilon
        sdf[ : in_surface_samples , :] = -curvature_radio  # in-surface = -epsilon

        # coordinates of the neighborhood of the tubular vicinity + off_surface
        coords = np.concatenate((in_surface_coords, on_surface_coords), axis=0)
        coords = np.concatenate((coords, out_surface_coords), axis=0)
        coords = np.concatenate((coords, off_surface_coords), axis=0)

        # duplicate the normals
        normals = np.concatenate((on_surface_normals, on_surface_normals), axis=0)
        normals = np.concatenate((normals, on_surface_normals), axis=0)
        normals = np.concatenate((normals, off_surface_normals), axis=0)

        min_curvature = np.concatenate((in_surface_min_curvature, on_surface_min_curvature))
        min_curvature = np.concatenate((min_curvature, out_surface_min_curvature))
        min_curvature = np.concatenate((min_curvature, off_surface_min_curvature))
        #min_curvature = np.expand_dims(min_curvature, -1)

        max_curvature = np.concatenate((in_surface_max_curvature, on_surface_max_curvature))
        max_curvature = np.concatenate((max_curvature, out_surface_max_curvature))
        max_curvature = np.concatenate((max_curvature, off_surface_max_curvature))
        #max_curvature = np.expand_dims(max_curvature, -1)

        return {'coords': torch.from_numpy(coords).float()}, {'sdf': torch.from_numpy(sdf).float(),
                                                              'normals': torch.from_numpy(normals).float(),
                                                              'min_curvature': torch.from_numpy(min_curvature).float(),
                                                              'max_curvature': torch.from_numpy(max_curvature).float()}


class PointCloudSDF(Dataset):
    """Point Cloud dataset with points sampled on the domain using their SDF
    value instead of -1 as per Sitzmann's work.

    Parameters
    ----------
    pointcloud_path: str
        Path to the input file. This file is loaded as a numpy array. We assume
        that the data is organized as follows: x, y, z, nx, ny, nz, k1, k2, sdf

    on_surface_points: int
        Number of points to fetch on the level-set 0 of the object.

    k_range: int, optional
        The maximum curvature value to allow. Any samples with absolute value
        of curvature larger than this will be REMOVED from the cloud. Default
        value is 10000.

    See Also
    --------
    numpy.genfromtxt
    """
    def __init__(self, pointcloud_path, on_surface_points, k_range=10000):
        super().__init__()

        print("Loading point cloud")
        point_cloud = np.genfromtxt(pointcloud_path)
        print("Finished loading point cloud")

        # exporting ply (point, curvatures, normal):  x, y, z, nx, ny, nz, k1, k2, sdf
        # Removing points with absurd curvatures.
        n = point_cloud.shape[0]
        point_cloud = point_cloud[np.absolute(point_cloud[:, 6]) <= k_range]
        out_min_k1 = n - point_cloud.shape[0]

        point_cloud = point_cloud[np.absolute(point_cloud[:, 7]) <= k_range]
        out_min_k2 = n - point_cloud.shape[0] + out_min_k1

        print(f"[WARN] Removed {out_min_k1} with abs(k1) > {k_range}.")
        print(f"[WARN] Removed {out_min_k2} with abs(k2) > {k_range}.")

        self.coords = point_cloud[:, :3]
        self.normals = point_cloud[:, 3:6]
        self.max_curvatures = point_cloud[:, 6]
        self.min_curvatures = point_cloud[:, 7]
        self.sdf = point_cloud[:, -1]

        self.on_surface_points = on_surface_points
        self.on_surface = self.sdf == 0

    def __len__(self):
        return self.coords.shape[0] // self.on_surface_points

    def __getitem__(self, idx):
        off_surface_samples = self.on_surface_points  # **2
        total_samples = self.on_surface_points + off_surface_samples

        # Random coords of on surface points
        on_surface_idx = np.flatnonzero(self.on_surface)
        rand_idcs = np.random.choice(on_surface_idx, size=self.on_surface_points)

        on_surface_coords = self.coords[rand_idcs, :]
        on_surface_normals = self.normals[rand_idcs, :]
        on_surface_min_curvature = self.min_curvatures[rand_idcs]
        on_surface_max_curvature = self.max_curvatures[rand_idcs]

        # Random coords of off-surface points
        off_surface_idx = np.flatnonzero(~self.on_surface)
        rand_idcs = np.random.choice(off_surface_idx, size=off_surface_samples)

        off_surface_coords = self.coords[rand_idcs, :]
        off_surface_normals = self.normals[rand_idcs, :]
        off_surface_min_curvature = self.min_curvatures[rand_idcs]
        off_surface_max_curvature = self.max_curvatures[rand_idcs]
        off_surface_sdf = self.sdf[rand_idcs]

        sdf = np.zeros((total_samples, 1))
        sdf[self.on_surface_points:, ] = off_surface_sdf[:, np.newaxis]

        coords = np.vstack((on_surface_coords, off_surface_coords))
        normals = np.vstack((on_surface_normals, off_surface_normals))

        min_curvature = np.concatenate((on_surface_min_curvature, off_surface_min_curvature))
        min_curvature = np.expand_dims(min_curvature, -1)
        max_curvature = np.concatenate((on_surface_max_curvature, off_surface_max_curvature))
        max_curvature = np.expand_dims(max_curvature, -1)

        return {'coords': torch.from_numpy(coords).float()}, {'sdf': torch.from_numpy(sdf).float(),
                                                              'normals': torch.from_numpy(normals).float(),
                                                              'min_curvature': torch.from_numpy(min_curvature).float(),
                                                              'max_curvature': torch.from_numpy(max_curvature).float()}


class PointCloudPrincipalDirections(Dataset):
    def __init__(self, pointcloud_path, on_surface_points, keep_aspect_ratio=True):
        super().__init__()

        print("Loading point cloud")
        point_cloud = np.genfromtxt(pointcloud_path)

        k_range = 5000
        n = point_cloud.shape[0]
        point_cloud = point_cloud[np.absolute(point_cloud[:, 3]) <= k_range]
        out_min_k1 = n - point_cloud.shape[0]

        point_cloud = point_cloud[np.absolute(point_cloud[:, 4]) <= k_range]
        out_min_k2 = n - point_cloud.shape[0] + out_min_k1

        print(f"[WARN] Removed {out_min_k1} with abs(k1) > {k_range}.")
        print(f"[WARN] Removed {out_min_k2} with abs(k2) > {k_range}.")

        print("Finished loading point cloud")

        #exporting ply (point, curvatures, normal, principal direction):  x, y, z, k1, k2, nx, ny, nz, kx, ky, kz
        coords = point_cloud[:, :3]
        self.min_curvatures = point_cloud[:, 4]
        self.max_curvatures = point_cloud[:, 3]
        self.normals = point_cloud[:, 5:8]
        self.principal_directions = point_cloud[:, 8:11]

        # Reshape point cloud such that it lies in bounding box of (-1, 1) (distorts geometry, but makes for high
        # sample efficiency)
        coords -= np.mean(coords, axis=0, keepdims=True)
        if keep_aspect_ratio:
            coord_max = np.amax(coords)
            coord_min = np.amin(coords)
        else:
            coord_max = np.amax(coords, axis=0, keepdims=True)
            coord_min = np.amin(coords, axis=0, keepdims=True)

        self.coords = (coords - coord_min) / (coord_max - coord_min)
        self.coords -= 0.5
        self.coords *= 2.

        self.on_surface_points = on_surface_points

    def __len__(self):
        return self.coords.shape[0] // self.on_surface_points

    def __getitem__(self, idx):
        point_cloud_size = self.coords.shape[0]

        off_surface_samples = self.on_surface_points  # **2
        total_samples = self.on_surface_points + off_surface_samples

        # Random coords
        rand_idcs = np.random.choice(point_cloud_size, size=self.on_surface_points)

        on_surface_coords = self.coords[rand_idcs, :]
        on_surface_normals = self.normals[rand_idcs, :]
        on_surface_min_curvature = self.min_curvatures[rand_idcs]
        on_surface_max_curvature = self.max_curvatures[rand_idcs]

        on_surface_principal_directions = self.principal_directions[rand_idcs, :]

        off_surface_coords = np.random.uniform(-1, 1, size=(off_surface_samples, 3))
        off_surface_normals = np.ones((off_surface_samples, 3)) * -1
        off_surface_min_curvature = np.zeros((off_surface_samples))
        off_surface_max_curvature = np.zeros((off_surface_samples))
        off_surface_principal_directions = np.ones((off_surface_samples, 3)) * -1

        # We consider the curvature of the sphere centered in the origin with radius equal to the norm of the coordinate.
        # off_surface_curvature = 1 / (np.linalg.norm(off_surface_coords, axis=1) ** 2)

        sdf = np.zeros((total_samples, 1))  # on-surface = 0
        sdf[self.on_surface_points:, :] = -1  # off-surface = -1

        coords = np.concatenate((on_surface_coords, off_surface_coords), axis=0)
        normals = np.concatenate((on_surface_normals, off_surface_normals), axis=0)
        min_curvature = np.concatenate((on_surface_min_curvature, off_surface_min_curvature))
        min_curvature = np.expand_dims(min_curvature, -1)
        max_curvature = np.concatenate((on_surface_max_curvature, off_surface_max_curvature))
        max_curvature = np.expand_dims(max_curvature, -1)
        principal_directions = np.concatenate((on_surface_principal_directions, off_surface_principal_directions), axis=0)

        return {'coords': torch.from_numpy(coords).float()}, {'sdf': torch.from_numpy(sdf).float(),
                                                              'normals': torch.from_numpy(normals).float(),
                                                              'min_curvature': torch.from_numpy(min_curvature).float(),
                                                              'max_curvature': torch.from_numpy(max_curvature).float(),
                                                              'principal_directions': torch.from_numpy(principal_directions).float()}


class PointCloudImplictFunctions(Dataset):
    def __init__(self, pointcloud_path, on_surface_points, keep_aspect_ratio=True):
        super().__init__()
        self.coords = np.random.uniform(-1, 1, size=(300000, 3))
        self.points = on_surface_points

    def __len__(self):
        return self.coords.shape[0] // self.points

    def __getitem__(self, idx):
        point_cloud_size = self.coords.shape[0]

        rand_idcs = np.random.choice(point_cloud_size, size=self.points)

        coords = torch.from_numpy(self.coords[rand_idcs, :]).float().unsqueeze(0)

        #function = implicit_functions.elipsoid()
        #function = implicit_functions.double_torus()
        function = implicit_functions.sdf_torus()
        function.eval()
        coord_values = function(coords)

        coords = coord_values['model_in']
        values = coord_values['model_out']#.unsqueeze(0)

        gradient = diff_operators.gradient(values, coords)#np.ones((off_surface_samples, 3)) * -1

        hessian = diff_operators.hessian(values, coords)
        min_curvature, max_curvature = diff_operators.principal_curvature(values, coords, gradient, hessian)
        principal_directions = diff_operators.principal_directions(gradient, hessian[0])[0]

        return {'coords': coords[0]}, {'sdf': values[0].cpu(),
                                       'normals': gradient[0].cpu(),
                                       'min_curvature': min_curvature.cpu(),
                                       'max_curvature': max_curvature.cpu(),
                                       'principal_directions': principal_directions[0].cpu()}


class PointCloudImplictFunctions_4D(Dataset):
    def __init__(self, pointcloud_path, on_surface_points, keep_aspect_ratio=True):
        super().__init__()
        self.coords = np.random.uniform(-1, 1, size=(100000, 3))
        self.points = on_surface_points

    def __len__(self):
        return self.coords.shape[0] // self.points

    def __getitem__(self, idx):
        point_cloud_size = self.coords.shape[0]

        time = torch.zeros((1, 3*self.points, 1))
        time[:, :self.points, :] = 0
        time[:, self.points:2*self.points, :] = 0.5
        time[:, 2*self.points:, :] = 1

        rand_idcs = np.random.choice(point_cloud_size, size=self.points)

        coords = torch.from_numpy(self.coords[rand_idcs, :]).float().unsqueeze(0)

        function1 = implicit_functions.elipsoid()
        function2 = implicit_functions.torus()
        function3 = implicit_functions.double_torus()
        function1.eval()
        function2.eval()
        function3.eval()
        coord_values1 = function1(coords)
        coord_values2 = function2(coords)
        coord_values3 = function3(coords)

        coords1 = coord_values1['model_in']
        values1 = coord_values1['model_out']#.unsqueeze(0)

        coords2 = coord_values2['model_in']
        values2 = coord_values2['model_out']#.unsqueeze(0)

        coords3 = coord_values3['model_in']
        values3 = coord_values3['model_out']#.unsqueeze(0)

        gradient1 = diff_operators.gradient(values1,coords1)#np.ones((off_surface_samples, 3)) * -1
        gradient2 = diff_operators.gradient(values2,coords2)#np.ones((off_surface_samples, 3)) * -1
        gradient3 = diff_operators.gradient(values3,coords3)#np.ones((off_surface_samples, 3)) * -1

        coords = torch.cat((coords1, coords2, coords3), axis=1)
        coords = torch.cat((coords, time), axis=-1)
        values = torch.cat((values1, values2, values3), axis=1)
        gradient = torch.cat((gradient1, gradient2, gradient3), axis=1)

        # print(coords)
        # print(coords.shape)
        # print(values.shape)
        # print(gradient.shape)

        return {'coords': coords[0]}, {'sdf': values[0].cpu(),
                                       'normals': gradient[0].cpu()}
