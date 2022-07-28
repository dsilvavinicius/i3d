all: logs/double_torus/checkpoints/model_current_weights.bin

clean:
	@rm -Rf logs/double_torus/
	@rm -Rf data/double_torus.xyz
	@rm -Rf __pycache__

data/double_torus.xyz: data/double_torus.ply
	@./tools/double_torus/preprocess_double_torus.sh

logs/double_torus/checkpoints/model_current.pth: data/double_torus.xyz
	@./tools/double_torus/train_double_torus.sh

logs/double_torus/checkpoints/model_current_weights.bin: logs/double_torus/checkpoints/model_current.pth
	@./tools/double_torus/double_torus_pth2bin.sh

comparison_analytic:
	python experiment_scripts/comparison_analytic.py --training_points 5000 --test_points 5000 --input sphere --fraction_on_surface 0.5 --methods rbf siren i3d i3dcurv --mc_resolution 128 --num_runs 100
	python experiment_scripts/comparison_analytic.py --training_points 5000 --test_points 5000 --input torus --fraction_on_surface 0.5 --methods rbf siren i3d i3dcurv --mc_resolution 128 --num_runs 100

comparison_ply_cuda:
	python experiment_scripts/comparison_ply.py --methods rbf siren i3d --resolution 256 --num_runs 10 --device cuda

comparison_ply:
	python experiment_scripts/comparison_ply.py --methods rbf siren i3d --resolution 256 --num_runs 10 --device cpu

.PHONY: all clean comparison_sphere comparison_torus comparison_ply_cuda comparison_ply train_models

train_release_models:
	python main.py experiments/armadillo_curvature_batch_sdf.json
	python main.py experiments/bunny_curvature_batch_sdf.json
	python main.py experiments/dragon_curvature_batch_sdf.json
	python main.py experiments/buddha_curvature_batch_sdf.json
	python main.py experiments/lucy_curvature_batch_sdf.json
