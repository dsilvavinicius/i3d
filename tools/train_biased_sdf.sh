#!/bin/bash

python experiment_scripts/train_curv_fracs.py --point_cloud_path=data/armadillo.ply --batch_size=10000 --experiment_name=armadillo_biased_b10000_w0-30_05-04-01_p70-p90 --num_epochs=1000 --w0=30

# python experiment_scripts/train_sdf_biased_curv.py --point_cloud_path=data/armadillo.ply --batch_size=10000 --experiment_name=armadillo_biased_sdf_curv_b10000_w0-30_e1000_06-02-02_p70-p95 --num_epochs=100 --w0=30
