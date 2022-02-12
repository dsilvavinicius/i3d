# coding: utf-8

import logging
from mesh_to_sdf.surface_point_cloud import SurfacePointCloud
from mesh_to_sdf import (get_surface_point_cloud, scale_to_unit_cube,
                         scale_to_unit_sphere)
import numpy as np
import trimesh
import torch
from torch.utils.data import Dataset


def _sample_on_surface(mesh: trimesh.Trimesh,
                       n_points: int,
                       sample_vertices=True) -> torch.Tensor:
    if sample_vertices:
        idx = np.random.choice(
            np.arange(start=0, stop=len(mesh.vertices)),
            size=n_points,
            replace=False
        )
        on_points = mesh.vertices[idx]
        on_normals = mesh.vertex_normals[idx]
    else:
        on_points, face_idx = mesh.sample(
            count=n_points,
            return_index=True
        )
        on_normals = mesh.face_normals[face_idx]

    return torch.from_numpy(np.hstack((
        on_points,
        on_normals,
        np.zeros((n_points, 1))
    )).astype(np.float32))


def _sample_domain(point_cloud: SurfacePointCloud,
                   n_points: int,
                   balance_in_out_points=False) -> torch.Tensor:
    domain_points = np.random.uniform(-1, 1, size=(n_points, 3))
    domain_sdf, domain_normals = point_cloud.get_sdf(
        domain_points,
        use_depth_buffer=False,
        return_gradients=True
    )
    if balance_in_out_points and n_points > 1:
        in_surf = domain_sdf < 0
        domain_sdf = np.concatenate((
            domain_sdf[in_surf],
            domain_sdf[~in_surf][:sum(in_surf)]), axis=0
        )
        domain_points = np.vstack((
            domain_points[in_surf, :],
            domain_points[~in_surf, :][:sum(in_surf)]
        ))
        domain_normals = np.vstack((
            domain_normals[in_surf, :],
            domain_normals[~in_surf, :][:sum(in_surf)]
        ))

    return torch.from_numpy(np.hstack((
        domain_points,
        domain_normals,
        domain_sdf[:, np.newaxis]
    )).astype(np.float32))


class PointCloud(Dataset):
    """SDF Point Cloud dataset.

    Parameters
    ----------
    mesh_path: str
        Path to the base mesh.

    samples_on_surface: int, optional
        Number of surface samples to fetch (i.e. {X | f(X) = 0}). Default value
        is None, meaning that all vertices will be used.

    scaling: str or None, optional
        The scaling to apply to the mesh. Possible values are: None
        (no scaling), "bbox" (-1, 1 in all axes), "sphere" (to fit the mesh in
        an unit sphere). Default is None.

    off_surface_sdf: number, optional
        Value to replace the SDF calculated by the sampling function for points
        with SDF != 0. May be used to replicate the behavior of Sitzmann et al.
        If set to `None` (default) uses the SDF estimated by the sampling
        function.

    off_surface_normals: np.array(size=(1, 3)), optional
        Value to replace the normals calculated by the sampling algorithm for
        points with SDF != 0. May be used to replicate the behavior of Sitzmann
        et al. If set to `None` (default) uses the SDF estimated by the
        sampling function.

    random_surf_samples: boolean, optional
        Wheter to randomly return points on surface (SDF=0) on `__getitem__`.
        If set to False (default), will return the specified point. Otherwise,
        `__getitem__` will return a randomly selected point, even if the same
        `idx` is provided.

    no_sampler: boolean, optional
        When this option is True, we assume that no sampler will be provided to
        the DataLoader, meaning that our `__getitem__` will return a batch of
        points instead of a single point, mimicking the behavior of Sitzmann
        et al. [1]. Default is False, meaning that an external sampler will be
        used.

    batch_size: integer, optional
        Only used when `no_sampler` is `True`. Used for fetching `batch_size`
        at every call of `__getitem__`. If set to 0 (default), fetches all
        on-surface points at every call.

    silent: boolean, optional
        Whether to report the progress of loading and processing the mesh (if
        set to False, default behavior), or not (if True).

    See Also
    --------
    trimesh.load, mesh_to_sdf.get_surface_point_cloud, _sample_domain,
    _sample_on_surface

    References
    ----------
    [1] Sitzmann, V., Martel, J. N. P., Bergman, A. W., Lindell, D. B.,
    & Wetzstein, G. (2020). Implicit Neural Representations with Periodic
    Activation Functions. ArXiv. Retrieved from http://arxiv.org/abs/2006.09661
    """
    def __init__(self, mesh_path, samples_on_surface=None, scaling=None,
                 off_surface_sdf=None, off_surface_normals=None,
                 random_surf_samples=False, no_sampler=False, batch_size=0,
                 silent=False):
        super().__init__()

        self.off_surface_sdf = off_surface_sdf
        self.no_sampler = no_sampler
        self.batch_size = batch_size

        if off_surface_normals is None:
            self.off_surface_normals = None
        else:
            self.off_surface_normals = torch.from_numpy(
                off_surface_normals.astype(np.float32)
            )

        if not silent:
            print(f"Loading mesh \"{mesh_path}\".")

        mesh = trimesh.load(mesh_path)
        if scaling is not None and scaling:
            if scaling == "bbox":
                mesh = scale_to_unit_cube(mesh)
            elif scaling == "sphere":
                mesh = scale_to_unit_sphere(mesh)
            else:
                raise ValueError("Invalid scaling option.")

        self.samples_on_surface = len(mesh.vertices)
        if samples_on_surface is None:
            print("Using *all* vertices as samples.")
        if samples_on_surface is not None:
            print(f"Using {samples_on_surface} vertices as samples.")
            self.samples_on_surface = samples_on_surface

        self.mesh = mesh
        if not silent:
            print("Creating point-cloud and acceleration structures.")

        if off_surface_sdf is None:
            self.point_cloud = get_surface_point_cloud(
                mesh,
                surface_point_method="scan",
                bounding_radius=1,
                calculate_normals=True
            )

        # We will fetch random samples at every access.
        self.random_surf_samples = random_surf_samples
        if not silent:
            print("Sampling surface.")

        self.surface_samples = _sample_on_surface(
            mesh,
            samples_on_surface,
            sample_vertices=True
        )

        if not silent:
            print("Done preparing the dataset.")

    def __len__(self):
        if self.no_sampler:
            return 2 * self.samples_on_surface // self.batch_size
        return self.samples_on_surface

    def __getitem__(self, idx):
        if self.no_sampler:
            return self._random_sampling(self.batch_size)

    def _random_sampling(self, n_points):
        """Randomly samples points on the surface and function domain."""
        if n_points <= 0:
            n_points = self.samples_on_surface

        on_surface_count = n_points // 2
        off_surface_count = n_points - on_surface_count

        on_surface_idx = np.random.choice(
            range(self.samples_on_surface),
            size=on_surface_count,
            replace=False
        )
        on_surface_samples = self.surface_samples[on_surface_idx, :]

        off_surface_points = np.random.uniform(-1, 1, size=(off_surface_count, 3))
        if self.off_surface_sdf is None:
            off_surface_sdf, off_surface_normals = self.point_cloud.get_sdf(
                off_surface_points,
                use_depth_buffer=False,
                return_gradients=True
            )
            off_surface_samples = torch.from_numpy(np.hstack((
                off_surface_points,
                off_surface_normals,
                off_surface_sdf[:, np.newaxis]
            )).astype(np.float32))
        else:
            off_surface_samples = torch.from_numpy(np.hstack((
                off_surface_points,
                np.full(shape=(off_surface_points.shape[0], 3), fill_value=-1),
                np.full(shape=(off_surface_points.shape[0], 1), fill_value=self.off_surface_sdf)
            )).astype(np.float32))
            # off_surface_samples[:, -1] = self.off_surface_sdf
            # off_surface_samples[:, 3:6] = self.off_surface_normals

        # if self.off_surface_sdf is not None:
        #     off_surface_samples[:, -1] = self.off_surface_sdf
        # if self.off_surface_normals is not None:
        #     off_surface_samples[:, 3:6] = self.off_surface_normals

        samples = torch.cat((on_surface_samples, off_surface_samples), dim=0)

        # Unsqueezing the SDF since it returns a shape [1] tensor and we need a
        # [1, 1] shaped tensor.
        return {
            "coords": samples[:, :3].float(),
        }, {
            "normals": samples[:, 3:6].float(),
            "sdf": samples[:, -1].unsqueeze(-1).float()
        }
