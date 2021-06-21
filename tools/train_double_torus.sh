#!/bin/bash
python experiment_scripts/train_sdf_preproc.py --model_type=sine --point_cloud_path=data/double_torus.xyz --batch_size=512 --experiment_name=home_double_torus_$(date --iso-8601=m) -w 5 --num_epochs=100
#python experiment_scripts/train_sdf_preproc.py --model_type=sine --point_cloud_path=data/double_torus.xyz --batch_size=512 --experiment_name=home_double_torus_$(date --iso-8601=m) -w 30 --num_epochs=100
#python experiment_scripts/train_sdf_preproc.py --model_type=sine --point_cloud_path=data/double_torus.xyz --batch_size=512 --experiment_name=home_double_torus_$(date --iso-8601=m) -w 60 --num_epochs=100
