# coding: utf-8

import math
import numpy as np
import open3d as o3d
import open3d.core as o3c
from plyfile import PlyData
import torch
from torch.utils.data import Dataset


def _sample_on_surface(mesh: o3d.t.geometry.TriangleMesh,
                       n_points: int,
                       exceptions: list = []) -> (torch.Tensor, np.ndarray):
    """Samples points from a mesh surface.

    Slightly modified from `i3d.dataset._sample_on_surface`. Returns the
    indices of points on surface as well and excludes points with indices in
    `exceptions`.

    Parameters
    ----------
    mesh: o3d.t.geometry.TriangleMesh
        The mesh to sample vertices from.

    n_points: int
        The number of vertices to sample.

    exceptions: list, optional
        The list of vertices to exclude from the selection. The default value
        is an empty list, meaning that any vertex might be selected. This works
        by setting the probabilities of any vertices with indices in
        `exceptions` to 0 and adjusting the probabilities of the remaining
        points.

    Returns
    -------
    samples: torch.Tensor
        The samples drawn from `mesh`

    idx: list
        The index of `samples` in `mesh`. Might be fed as input to further
        calls of `sample_on_surface`

    See Also
    --------
    numpy.random.choice
    """
    if exceptions:
        p = np.array(
            [1. / (len(mesh.vertex["positions"]) - len(exceptions))] *
            len(mesh.vertex["positions"])
        )
        p[exceptions] = 0.0

    idx = np.random.choice(
        np.arange(start=0, stop=len(mesh.vertex["positions"])),
        size=n_points,
        replace=False,
        p=p if exceptions else None
    )
    on_points = mesh.vertex["positions"].numpy()[idx]
    on_normals = mesh.vertex["normals"].numpy()[idx]

    return torch.from_numpy(np.hstack((
        on_points,
        on_normals,
        np.zeros((n_points, 1))
    )).astype(np.float32)), idx.tolist()


def _lowMedHighCurvSegmentation(
        mesh: o3d.t.geometry.TriangleMesh,
        n_samples: int,
        bin_edges: np.array,
        proportions: np.array,
        exceptions: list = []
):
    """Samples `n_points` points from `mesh` based on their curvature.

    This function is based on `i3d.dataset.lowMedHighCurvSegmentation`.

    Parameters
    ----------
    mesh: o3d.t.geometry.TriangleMesh,
        The mesh to sample points from.

    n_samples: int
        Number of samples to fetch.

    bin_edges: np.array
        The [minimum, low-medium threshold, medium-high threshold, maximum]
        curvature values in `mesh`. These values define thresholds between low
        and medium curvature values, and medium to high curvatures.

    proportions: np.array
        The percentage of points to fetch for each curvature band per batch of
        `n_samples`.

    Returns
    -------
    samples: torch.Tensor
        The vertices sampled from `mesh`.
    """
    on_surface_sampled = 0
    on_surface_pts = torch.column_stack((
        torch.from_numpy(mesh.vertex["positions"].numpy()),
        torch.from_numpy(mesh.vertex["normals"].numpy()),
        torch.from_numpy(mesh.vertex["curvatures"].numpy())
    ))

    if exceptions:
        index = torch.Tensor(
            list(set(range(on_surface_pts.shape[0])) - set(exceptions)),
        ).int()
        on_surface_pts = torch.index_select(
            on_surface_pts, dim=0, index=index
        )

    curvatures = on_surface_pts[..., -1]

    n_low_curvature = int(math.floor(proportions[0] * n_samples))
    low_curvature_pts = on_surface_pts[(curvatures >= bin_edges[0]) & (curvatures < bin_edges[1]), ...]
    low_curvature_idx = np.random.choice(
        range(low_curvature_pts.shape[0]),
        size=n_low_curvature,
        replace=True if n_low_curvature > low_curvature_pts.shape[0] else False
    )
    on_surface_sampled = len(low_curvature_idx)

    n_med_curvature = int(math.ceil(proportions[1] * n_samples))
    med_curvature_pts = on_surface_pts[(curvatures >= bin_edges[1]) & (curvatures < bin_edges[2]), ...]
    med_curvature_idx = np.random.choice(
        range(med_curvature_pts.shape[0]),
        size=n_med_curvature,
        replace=True if n_med_curvature > med_curvature_pts.shape[0] else False
    )
    on_surface_sampled += len(med_curvature_idx)

    n_high_curvature = n_samples - on_surface_sampled
    high_curvature_pts = on_surface_pts[(curvatures >= bin_edges[2]) & (curvatures <= bin_edges[3]), ...]
    high_curvature_idx = np.random.choice(
        range(high_curvature_pts.shape[0]),
        size=n_high_curvature,
        replace=True if n_high_curvature > high_curvature_pts.shape[0] else False
    )

    return torch.cat((
        low_curvature_pts[low_curvature_idx, ...],
        med_curvature_pts[med_curvature_idx, ...],
        high_curvature_pts[high_curvature_idx, ...]
    ), dim=0)


def _read_ply_with_curvatures(path: str):
    """Reads a PLY file with position, normal and curvature info.

    Note that we expect the input ply to contain x,y,z vertex data, as well
    as nx,ny,nz normal data and the curvature stored in the `quality` vertex
    property.

    Parameters
    ----------
    path: str, PathLike
        Path to the ply file. We except the file to be in binary format.

    Returns
    -------
    mesh: o3d.t.geometry.TriangleMesh
        The fully constructed Open3D Triangle Mesh. By default, the mesh is
        allocated on the CPU:0 device.

    vertices: numpy.array
        The same vertex information as stored in `mesh` returned for
        convenience only.

    See Also
    --------
    PlyData.read, o3d.t.geometry.TriangleMesh
    """
    # Reading the PLY file with curvature info
    with open(path, "rb") as f:
        plydata = PlyData.read(f)
        num_verts = plydata["vertex"].count
        vertices = np.zeros(shape=(num_verts, 7), dtype=np.float32)
        vertices[:, 0] = plydata["vertex"].data["x"]
        vertices[:, 1] = plydata["vertex"].data["y"]
        vertices[:, 2] = plydata["vertex"].data["z"]
        vertices[:, 3] = plydata["vertex"].data["nx"]
        vertices[:, 4] = plydata["vertex"].data["ny"]
        vertices[:, 5] = plydata["vertex"].data["nz"]
        vertices[:, 6] = plydata["vertex"].data["quality"]

        faces = np.stack(plydata["face"].data["vertex_indices"])

    # Converting the PLY data to open3d format
    device = o3c.Device("CPU:0")
    mesh = o3d.t.geometry.TriangleMesh(device)
    mesh.vertex["positions"] = o3c.Tensor(vertices[:, :3], dtype=o3c.float32)
    mesh.vertex["normals"] = o3c.Tensor(vertices[:, 3:6], dtype=o3c.float32)
    mesh.vertex["curvatures"] = o3c.Tensor(vertices[:, -1], dtype=o3c.float32)
    mesh.triangle["indices"] = o3c.Tensor(faces, dtype=o3c.int32)

    return mesh, vertices


def _create_training_data(
        mesh: o3d.t.geometry.TriangleMesh,
        n_on_surf: int,
        n_off_surf: int,
        on_surface_exceptions: list = [],
        domain_bounds: tuple = ([-1, -1, -1], [1, 1, 1]),
        scene: o3d.t.geometry.RaycastingScene = None,
        no_sdf: bool = False,
        use_curvature: bool = False,
        curvature_fractions: list = [],
        curvature_thresholds: list = [],
):
    """Creates a set of training data with coordinates, normals and SDF
    values.

    Parameters
    ----------
    mesh: o3d.t.geometry.TriangleMesh
        A Tensor-backed Open3D mesh.

    n_on_surf: int
        # of points to sample from the mesh.

    n_off_surf: int
        # of points to sample from the domain. Note that we sample points
        uniformely at random from the domain.

    on_surface_exceptions: list, optional
        Points that cannot be used for training, i.e. test set of points.

    domain_bounds: tuple[np.array, np.array]
        Bounds to use when sampling points from the domain.

    scene: o3d.t.geometry.RaycastingScene
        Open3D raycasting scene to use when querying SDF for domain points.

    no_sdf: boolean, optional
        If using SIREN's original loss, we do not query SDF for domain
        points, instead we mark them with SDF = -1.

    use_curvature: boolean, optional
        Indicates if we must use the curvature to perform sampling on surface
        points.

    curvature_fractions: list, optional
        The fractions of points to sample per curvature band. Only used when
        `use_curvature` is True.

    curvature_thresholds: list
        The curvature values to use when defining low, medium and high
        curvatures. Only used when `use_curvature` is True.

    Returns
    -------
    full_pts: torch.Tensor
    full_normals: torch.Tensor
    full_sdf: torch.Tensor

    See Also
    --------
    _sample_on_surface, _lowMedHighCurvSegmentation
    """
    if use_curvature:
        surf_pts = _lowMedHighCurvSegmentation(
            mesh,
            n_on_surf,
            curvature_thresholds,
            curvature_fractions,
            on_surface_exceptions
        )
    else:
        surf_pts, _ = _sample_on_surface(
            mesh,
            n_on_surf,
            on_surface_exceptions
        )
    surf_pts = torch.from_numpy(surf_pts.numpy())

    domain_pts = np.random.uniform(
        domain_bounds[0], domain_bounds[1],
        (n_off_surf, 3)
    )

    if not no_sdf:
        domain_pts = o3c.Tensor(domain_pts, dtype=o3c.Dtype.Float32)
        domain_sdf = scene.compute_signed_distance(domain_pts)
        domain_sdf = torch.from_numpy(domain_sdf.numpy())
        domain_pts = torch.from_numpy(domain_pts.numpy())
    else:
        domain_pts = torch.from_numpy(domain_pts)
        domain_sdf = -1 * torch.ones(domain_pts.shape[0])

    domain_normals = torch.zeros_like(domain_pts)

    full_pts = torch.row_stack((
        surf_pts[..., :3],
        domain_pts
    ))
    full_normals = torch.row_stack((
        surf_pts[..., 3:6],
        domain_normals
    ))
    full_sdf = torch.cat((
        torch.zeros(len(surf_pts)),
        domain_sdf
    ))

    return full_pts.float(), full_normals.float(), full_sdf.float()


def _calc_curvature_bins(curvatures: torch.Tensor, percentiles: list):
    """Bins the curvature values according to `percentiles`.

    Parameters
    ----------
    curvatures: torch.Tensor
        Tensor with the curvature values for the vertices.

    percentiles: list
        List with the percentiles.
    """
    q = torch.quantile(curvatures, torch.Tensor(percentiles) / 100.0)
    return [
        curvatures.min().item(),
        q[0].item(),
        q[1].item(),
        curvatures.max().item()
    ]


class PointCloud(Dataset):
    """SDF Point Cloud dataset.

    Parameters
    ----------
    mesh_path: str
        Path to the base mesh.

    n_on_surface: int, optional
        Number of surface samples to fetch (i.e. {X | f(X) = 0}). Default value
        is None, meaning that all vertices will be used.

    off_surface_sdf: number, optional
        Value to replace the SDF calculated by the sampling function for points
        with SDF != 0. May be used to replicate the behavior of Sitzmann et al.
        If set to `None` (default) uses the SDF estimated by the sampling
        function.

    off_surface_normals: torch.Tensor, optional
        Value to replace the normals calculated by the sampling algorithm for
        points with SDF != 0. May be used to replicate the behavior of Sitzmann
        et al. If set to `None` (default) uses the SDF estimated by the
        sampling function.

    batch_size: integer, optional
        Used for fetching `batch_size` at every call of `__getitem__`. If set
        to 0 (default), fetches all on-surface points at every call.

    use_curvature: boolean, optional
        Indicates if we must use the curvature to perform sampling on surface
        points. By default this is False.

    curvature_fractions: list, optional
        The fractions of points to sample per curvature band. Only used when
        `use_curvature` is True.

    curvature_percentiles: list, optional
        The curvature percentiles to use when defining low, medium and high
        curvatures. Only used when `use_curvature` is True.

    References
    ----------
    [1] Sitzmann, V., Martel, J. N. P., Bergman, A. W., Lindell, D. B.,
    & Wetzstein, G. (2020). Implicit Neural Representations with Periodic
    Activation Functions. ArXiv. Retrieved from http://arxiv.org/abs/2006.09661
    """
    def __init__(self, mesh_path: str,
                 batch_size: int,
                 off_surface_sdf: float = None,
                 off_surface_normals: torch.Tensor = None,
                 use_curvature: bool = False,
                 curvature_fractions: list = [],
                 curvature_percentiles: list = []):
        super().__init__()

        self.off_surface_normals = None
        if off_surface_normals is not None:
            if isinstance(off_surface_normals, list):
                self.off_surface_normals = torch.Tensor(off_surface_normals)

        print(f"Loading mesh \"{mesh_path}\".")
        print("Using curvatures? ", "YES" if use_curvature else "NO")
        if use_curvature:
            self.mesh, _ = _read_ply_with_curvatures(mesh_path)
        else:
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            mesh.compute_vertex_normals()
            self.mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

        self.batch_size = batch_size
        print(f"Fetching {self.batch_size // 2} on-surface points per iteration.")

        print("Creating point-cloud and acceleration structures.")
        self.off_surface_sdf = off_surface_sdf
        self.scene = None
        if off_surface_sdf is None:
            self.scene = o3d.t.geometry.RaycastingScene()
            self.scene.add_triangles(self.mesh)

        self.use_curvature = use_curvature
        self.curvature_fractions = curvature_fractions
        self.mesh_size = len(self.mesh.vertex["positions"])

        # Binning the curvatures
        self.curvature_bins = None
        if use_curvature:
            self.curvature_bins = _calc_curvature_bins(
                torch.from_numpy(self.mesh.vertex["curvatures"].numpy()),
                curvature_percentiles
            )
        print("Done preparing the dataset.")

    def __len__(self):
        return 2 * self.mesh_size // self.batch_size

    def __getitem__(self, idx):
        pts, normals, sdf = _create_training_data(
            mesh=self.mesh,
            n_on_surf=self.batch_size // 2,
            n_off_surf=self.batch_size // 2,
            scene=self.scene,
            no_sdf=self.off_surface_sdf is not None,
            use_curvature=self.use_curvature,
            curvature_fractions=self.curvature_fractions,
            curvature_thresholds=self.curvature_bins
        )
        return {
            "coords": pts.float(),
        }, {
            "normals": normals.float(),
            "sdf": sdf.unsqueeze(1).float()
        }


if __name__ == "__main__":
    p = PointCloud(
        "data/armadillo_curvs.ply", batch_size=10, use_curvature=True,
        curvature_fractions=(0.2, 0.7, 0.1), curvature_percentiles=(70, 95)
    )
    print(len(p))
    print(p.__getitem__(0))
    print(p.__getitem__(0))
    print(p.__getitem__(0))

    p = PointCloud("data/armadillo.ply", batch_size=10, use_curvature=False)
    print(len(p))
    print(p.__getitem__(0))
    print(p.__getitem__(0))
    print(p.__getitem__(0))
