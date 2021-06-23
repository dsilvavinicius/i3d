# coding: utf-8

import sys
import os
from torch.utils.data import DataLoader
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataio
import meta_modules
import utils
import training
import loss_functions
import modules


class CodeCoverageTest(unittest.TestCase):
    def setUp(self):
        self.logging_root = "../logs"
        self.experiment_name = "CodeCoverageTest"

        self.batch_size = 4096
        self.lr = 1e-4
        self.num_epochs = 100

        self.epochs_til_ckpt = 1
        self.steps_til_summary = 100

        self.model_type = "sine"
        self.point_cloud_path = "../data/double_torus.xyz"
        self.checkpoint_path = None

    def test_model_training(self):
        sdf_dataset = dataio.PointCloudSDF(
            self.point_cloud_path,
            on_surface_points=self.batch_size
        )

        dataloader = DataLoader(
            sdf_dataset,
            shuffle=True,
            batch_size=1,
            pin_memory=True,
            num_workers=0
        )

        model = modules.SingleBVPNet(type=self.model_type, in_features=3)
        loss_fn = loss_functions.sdf_on_off_surf
        summary_fn = utils.write_sdf_summary

        root_path = os.path.join(self.logging_root, self.experiment_name)

        training.train(model, dataloader, self.num_epochs, self.lr,
                       self.steps_til_summary, self.epochs_til_ckpt,
                       root_path, loss_fn, summary_fn,
                       double_precision=False, clip_grad=False)


if __name__ == "__main__":
    unittest.main()
