.PHONY: all clean comparison_analytic comparison_ply_cuda comparison_ply train_release_models

all: logs/double_torus/checkpoints/model_current_weights.bin

clean:
	@rm -Rf logs/double_torus/
	@rm -Rf data/double_torus.xyz
	@rm -Rf __pycache__

comparison_analytic:
	python experiment_scripts/comparison_analytic.py --training_points 5000 --test_points 5000 --input sphere --fraction_on_surface 0.5 --methods rbf siren i3d i3dcurv --mc_resolution 128 --num_runs 100
	python experiment_scripts/comparison_analytic.py --training_points 5000 --test_points 5000 --input torus --fraction_on_surface 0.5 --methods rbf siren i3d i3dcurv --mc_resolution 128 --num_runs 100

comparison_ply_cuda:
	python experiment_scripts/comparison_ply.py --methods rbf siren i3d --resolution 256 --num_runs 10 --device cuda

comparison_ply:
	python experiment_scripts/comparison_ply.py --methods rbf siren i3d --resolution 256 --num_runs 10 --device cpu

train_release_models:
	python main.py experiments/armadillo_curvature_batch_sdf.json
	python main.py experiments/bunny_curvature_batch_sdf.json
	python main.py experiments/dragon_curvature_batch_sdf.json
	python main.py experiments/buddha_curvature_batch_sdf.json
	python main.py experiments/lucy_curvature_batch_sdf.json
