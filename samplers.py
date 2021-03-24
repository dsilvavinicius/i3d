# coding: utf-8

from torch.utils.data import Sampler


class CurvatureSeqSampler(Sampler):
    """A simple sampler that orders the data samples by curvature and returns
    them in sequence.

    Parameters
    ----------
    data_source: PointCloud
        The actual data source. Must have the curvatures as features.
    """
    def __init__(self, data_source):
        self.data_source = data_source
        self.sorted_curvatures_idx = np.argsort(np.absolute(self.data_source.curvatures))

    def __iter__(self):
        return iter(range(len(self.sorted_curvatures_idx)))

    def __len__(self):
        return len(self.data_source)


class CurvatureWeightedSampler(Sampler):
    """Sampler that orders the data samples by absolute value of curvature and returns
    them given a probability.

    Parameters
    ----------
    data_source: PointCloud
        The actual data source. Must have the curvatures as features.
    """
    def __init__(self, data_source):
        self.data_source = data_source
        self.sorted_curvatures_idx = np.argsort(np.absolute(self.data_source.curvatures))

    def __iter__(self):
        return iter(range(len(self.sorted_curvatures_idx)))

    def __len__(self):
        return len(self.data_source)
