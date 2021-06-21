#!/bin/bash
python experiment_scripts/train_sdf_biased.py --model_type=sine --point_cloud_path=data/test_big_4.xyz --batch_size=2048 --experiment_name=home_biasedsampler_$(date --iso-8601=m) --num_epochs=100
