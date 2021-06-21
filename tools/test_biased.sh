#!/bin/bash
python experiment_scripts/test_sdf.py --checkpoint_path=logs/home_biasedsampler_2021-03-28T23:08-03:00/checkpoints/model_epoch_0001.pth --experiment_name=home_biased_sampler_1_rec_30_old --resolution=512
python experiment_scripts/test_sdf.py --checkpoint_path=logs/home_biasedsampler_2021-03-28T23:08-03:00/checkpoints/model_epoch_0010.pth --experiment_name=home_biased_sampler_10_rec_30_old --resolution=512
python experiment_scripts/test_sdf.py --checkpoint_path=logs/home_biasedsampler_2021-03-28T23:08-03:00/checkpoints/model_epoch_0050.pth --experiment_name=home_biased_sampler_50_rec_30_old --resolution=512
python experiment_scripts/test_sdf.py --checkpoint_path=logs/home_biasedsampler_2021-03-28T23:08-03:00/checkpoints/model_final.pth --experiment_name=home_biased_sampler_100_rec_30_old --resolution=512

python experiment_scripts/test_sdf.py --checkpoint_path=logs/home_biasedsampler_2021-03-29T09:37-03:00/checkpoints/model_epoch_0001.pth --experiment_name=home_biased_sampler_1_rec_30 --resolution=512
python experiment_scripts/test_sdf.py --checkpoint_path=logs/home_biasedsampler_2021-03-29T09:37-03:00/checkpoints/model_epoch_0010.pth --experiment_name=home_biased_sampler_10_rec_30 --resolution=512
python experiment_scripts/test_sdf.py --checkpoint_path=logs/home_biasedsampler_2021-03-29T09:37-03:00/checkpoints/model_epoch_0050.pth --experiment_name=home_biased_sampler_50_rec_30 --resolution=512
python experiment_scripts/test_sdf.py --checkpoint_path=logs/home_biasedsampler_2021-03-29T09:37-03:00/checkpoints/model_final.pth --experiment_name=home_biased_sampler_100_rec_30 --resolution=512
