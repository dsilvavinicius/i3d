# coding: utf-8

from torch.utils.data import Sampler
import numpy as np


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
        curvatures = (self.data_source.min_curvatures + self.data_source.max_curvatures) / 2
        self.sorted_curvatures_idx = np.argsort(np.absolute(curvatures))

    def __iter__(self):
        return iter(range(len(self.sorted_curvatures_idx)))

    def __len__(self):
        return len(self.data_source)


class CurvatureHistogramSampler(Sampler):
    """Sampler that orders the data samples by absolute value of curvature and
    returns them given a probability.

    Parameters
    ----------
    data_source: PointCloud
        The actual data source. Must have the curvatures as features.
    """
    def __init__(self, data_source, bins=10):
        self.data_source = data_source
        mean_curvatures = (self.data_source.min_curvatures + self.data_source.max_curvatures) / 2
        hist = np.histogram(mean_curvatures, bins=bins)
        self.weights = hist / sum(hist)
        self.sorted_curvatures_idx = np.argsort(np.absolute(mean_curvatures))

    def __iter__(self):
        return iter(range(len(self.sorted_curvatures_idx)))

    def __len__(self):
        return len(self.data_source)

    def update_weights(self, weights):
        pass
