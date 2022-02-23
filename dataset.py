# coding: utf-8

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

    no_sampler: boolean, optional
        When this option is True, we assume that no sampler will be provided to
        the DataLoader, meaning that our `__getitem__` will return a batch of
        points instead of a single point, mimicking the behavior of Sitzmann
        et al. [1]. Default is True, meaning that no external sampler will be
        used.

    batch_size: integer, optional
        Only used when `no_sampler` is `True`. Used for fetching `batch_size`
        at every call of `__getitem__`. If set to 0 (default), fetches all
        on-surface points at every call.

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
                 no_sampler=True, batch_size=0):
        super().__init__()

        self.off_surface_sdf = off_surface_sdf
        self.no_sampler = no_sampler

        if off_surface_normals is None:
            self.off_surface_normals = None
        else:
            if isinstance(off_surface_normals, list):
                off_surface_normals = np.array(off_surface_normals)
            self.off_surface_normals = torch.from_numpy(
                off_surface_normals.astype(np.float32)
            )

        print(f"Loading mesh \"{mesh_path}\".")
        mesh = trimesh.load(mesh_path)

        self.samples_on_surface = len(mesh.vertices)
        if not samples_on_surface:
            print("Using *all* vertices as samples.")
        else:
            print(f"Using {samples_on_surface} vertices as samples.")
            self.samples_on_surface = samples_on_surface

        self.batch_size = batch_size
        if not batch_size:
            self.batch_size = self.samples_on_surface
        print(f"Fetching {self.batch_size} on-surface points per iteration.")

        print("Mesh scaling:", scaling)
        if scaling is not None and scaling:
            if scaling == "bbox":
                mesh = scale_to_unit_cube(mesh)
            else:
                raise ValueError("Invalid scaling option.")

        self.mesh = mesh
        print("Creating point-cloud and acceleration structures.")

        if off_surface_sdf is None:
            self.point_cloud = get_surface_point_cloud(
                mesh,
                surface_point_method="scan",
                calculate_normals=True
            )

        print("Sampling surface.")
        self.surface_samples = _sample_on_surface(
            mesh,
            self.samples_on_surface,
            sample_vertices=True
        )

        print("Done preparing the dataset.")

    def __len__(self):
        if self.no_sampler:
            return self.samples_on_surface // self.batch_size
        return self.samples_on_surface

    def __getitem__(self, idx):
        if self.no_sampler:
            return self._random_sampling(self.batch_size)

    def _random_sampling(self, n_points):
        """Randomly samples points on the surface and function domain."""
        if n_points <= 0:
            n_points = self.surface_samples.shape[0]

        on_surface_count = n_points
        off_surface_count = n_points

        on_surface_idx = np.random.choice(
            range(self.surface_samples.shape[0]),
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
        else:
            off_surface_normals = np.full(
                shape=(off_surface_count, 3),
                fill_value=-1
            )
            off_surface_sdf = np.full(
                shape=(off_surface_count, 1),
                fill_value=self.off_surface_sdf
            ).squeeze()

        off_surface_samples = torch.from_numpy(np.hstack((
            off_surface_points,
            off_surface_normals,
            off_surface_sdf[:, np.newaxis]
        )).astype(np.float32))

        samples = torch.cat((on_surface_samples, off_surface_samples), dim=0)

        # Unsqueezing the SDF since it returns a shape [1] tensor and we need a
        # [1, 1] shaped tensor.
        return {
            "coords": samples[:, :3].float(),
        }, {
            "normals": samples[:, 3:6].float(),
            "sdf": samples[:, -1].unsqueeze(-1).float()
        }


class PointCloudSDFPreComputedCurvaturesDirections(Dataset):
    """Data class of a point-cloud that calculates the SDF values of point
    samples and schedules the samples by their curvatures.

    Parameters
    ----------
    mesh_path: str

    low_med_percentiles: collection[numbers], optional

    curvature_func: function(trimesh.Mesh, list[points], number), optional

    curvature_fracs: collection[numbers], optional

    scaling: str, optional

    uniform_sampling: boolean, optional

    batch_size: int, optional

    silent: boolean, optional

    See Also
    --------
    trimesh.curvature.discrete_gaussian_curvature_measure,
    trimesh.curvature.discrete_mean_curvature_measure
    """
    def __init__(self, mesh_path, xyz_path, low_med_percentiles=(70, 95),
                 curvature_fracs=(0.2, 0.6, 0.2), scaling=None,
                 uniform_sampling=False, batch_size=0, silent=False):
        super().__init__()
        self.uniform_sampling = uniform_sampling
        self.low_med_percentiles = low_med_percentiles

        # Loading the curvatures 
        print("Loading xyz point cloud")
        point_cloud_curv = np.genfromtxt(xyz_path)
        #point_cloud_curv = np.genfromtxt('./data/armadillo_curv_dir.xyz')
        #remove nan
        point_cloud_curv = point_cloud_curv[~np.isnan(point_cloud_curv[:, 4])] 
        print("Finished loading point cloud")

        #exporting ply (point, curvatures, normal, principal direction):  x, y, z, k1, k2, nx, ny, nz, kx, ky, kz       
        #coords = point_cloud[:, :3]
        min_curvatures = point_cloud_curv[:, 4]# the signal was changed
        max_curvatures = point_cloud_curv[:, 3]
        normals = point_cloud_curv[:, 5:8] #we should use mesh.vertex_normals because the sdf is computed using it
        max_dirs = point_cloud_curv[:, 8:11]

        #Loading the mesh
        self.input_path = mesh_path
        self.batch_size = batch_size

        print(f"Loading mesh \"{mesh_path}\".")

        mesh = trimesh.load(mesh_path)
        if scaling is not None and scaling:
            if scaling == "bbox":
                mesh = scale_to_unit_cube(mesh)
            else:
                raise ValueError("Invalid scaling option.")

        self.mesh = mesh
        print("Creating point-cloud and acceleration structures.")

        self.point_cloud = get_surface_point_cloud(mesh, surface_point_method="scan", calculate_normals=True )

        #self.diff_curvatures = np.abs(min_curvatures-max_curvatures)
        self.gauss_curvatures = min_curvatures*max_curvatures
        #self.abs_curvatures = 0.5*np.abs(min_curvatures+max_curvatures)
        self.abs_curvatures = np.abs(min_curvatures)+np.abs(max_curvatures)

        #using harris corner detector
        #self.abs_curvatures = min_curvatures*max_curvatures - 0.05*(min_curvatures+max_curvatures)**2

        # planar, edge, corner region fractions
        #self.curvature_fracs = curvature_fracs
        
        # low, medium, high curvature fractions
        self.curvature_fracs = curvature_fracs
        l1, l2 = np.percentile(self.abs_curvatures, low_med_percentiles)
        self.bin_edges = [
            np.min(self.abs_curvatures),
            l1,
            l2,
            np.max(self.abs_curvatures)
        ]

        #bbox scaling
        vertices = point_cloud_curv[:, 0:3] - mesh.bounding_box.centroid
        vertices *= 2 / np.max(mesh.bounding_box.extents)       

        self.surface_samples = torch.from_numpy(np.hstack((
            vertices,
            normals,
            #min_curvatures[:, np.newaxis],
            #max_curvatures[:, np.newaxis],
            #max_dirs,
            np.zeros((len(vertices), 1))
        )).astype(np.float32))

        if not silent:
            print("Done preparing the dataset.")

    def __len__(self):
    #    return self.surface_samples.size(0) // self.batch_size

        lenght = self.surface_samples.size(0) // (self.batch_size) + 1
        # lenght = 17
        return lenght

        # # percentile of med curv samples times 1.2 used when we are using the whole dataset
        # p2o = 1.2*(self.low_med_percentiles[1]-self.low_med_percentiles[0])/100
        
        # # percentile of med curv samples times 1.2 used when we are using a percentile p2o/p2 of dataset
        # p2 = self.curvature_fracs[1]

        # lenght = int(math.floor((p2o/p2)*self.surface_samples.size(0))) // self.batch_size
        # return lenght
        # #return 100000 // self.batch_size
        #return 50000 // self.batch_size
        #return 10000 // self.batch_size

    def __getitem__(self, idx):
        return self._random_sampling(self.batch_size)

    def _random_sampling(self, n_points):
        """Randomly samples points on the surface and function domain."""
        if n_points <= 0:
            n_points = self.surface_samples.size(0)

        on_surface_count = n_points
        off_surface_count = n_points

        on_surface_samples = []

        if self.uniform_sampling:
            idx = np.random.choice(
                self.surface_samples.size(0),
                on_surface_count,
                replace=False
            )
            on_surface_samples = self.surface_samples[idx, ...]
        else:
            #on_surface_samples=edgePlanarCornerSegmentation(self.surface_samples, on_surface_count, self.abs_curvatures, self.bin_edges, self.curvature_fracs)
            on_surface_samples = lowMedHighCurvSegmentation(self.surface_samples, on_surface_count, self.abs_curvatures, self.bin_edges, self.curvature_fracs)

        off_surface_points = np.random.uniform(-1, 1, size=(off_surface_count, 3))
        off_surface_sdf, off_surface_normals = self.point_cloud.get_sdf(
            off_surface_points,
            use_depth_buffer=False,
            return_gradients=True
        )
        # print(off_surface_sdf.shape, off_surface_normals.shape)

        off_surface_samples = torch.from_numpy(np.hstack((
            off_surface_points,
            off_surface_normals,
            #np.zeros((off_surface_count, 1)),#min_curv
            #np.zeros((off_surface_count, 1)),#max_curv
            #np.ones((off_surface_count, 3)) * -1,#max_dirs
            off_surface_sdf[:, np.newaxis]
        )).astype(np.float32))
        #off_surface_samples[:, 3:6] = -1
        #off_surface_samples[:, -1] = -1

        samples = torch.cat((on_surface_samples, off_surface_samples), dim=0)
        print(samples.shape)
        print(samples[:5, :])
        print(samples[-5:, :])

        # Unsqueezing the SDF since it returns a shape [1] tensor and we need a
        # [1, 1] shaped tensor.
        return {
            "coords": samples[:, :3].float()
        }, {
            "normals": samples[:, 3:6].float(),
            "sdf": samples[:, -1].unsqueeze(-1).float()
        }
