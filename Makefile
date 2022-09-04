.PHONY: all clean comparison_analytic comparison_ply_cuda comparison_ply train_release_models

all: logs/double_torus/checkpoints/model_current_weights.bin

clean:
	rm -Rf logs/double_torus/
	rm -Rf data/double_torus.xyz
	rm -Rf __pycache__

comparison_analytic:
	python experiment_scripts/comparison_analytic.py --training_points 5000 --test_points 5000 --input sphere --fraction_on_surface 0.5 --methods rbf siren i3d i3dcurv --mc_resolution 128 --num_runs 100
	python experiment_scripts/comparison_analytic.py --training_points 5000 --test_points 5000 --input torus --fraction_on_surface 0.5 --methods rbf siren i3d i3dcurv --mc_resolution 128 --num_runs 100

comparison_ply_cuda:
	python experiment_scripts/comparison_ply.py --methods rbf siren i3d --resolution 256 --num_runs 10 --device cuda

comparison_ply:
	python experiment_scripts/comparison_ply.py --methods rbf siren i3d --resolution 256 --num_runs 10 --device cpu

train_release_models: data/armadillo_curvs.ply data/bunny_curvs.ply data/dragon_curvs.ply data/happy_buddha_curvs.ply data/lucy_simple_curvs.ply
	python main.py experiments/armadillo_curvature_batch_sdf.json
	python main.py experiments/bunny_curvature_batch_sdf.json
	python main.py experiments/dragon_curvature_batch_sdf.json
	python main.py experiments/buddha_curvature_batch_sdf.json
	python main.py experiments/lucy_curvature_batch_sdf.json

data/armadillo_curvs.ply:
	@mkdir -p data
	./download_data.sh 14y6MxrOPCLR2yggjjdYnvD9VAljhh024 $@
	@rm -f cookie

data/bunny_curvs.ply:
	@mkdir -p data
	./download_data.sh 1gKYPtQ2hvAdjjKsDEJPQvuczLOm29d9o $@
	@rm -f cookie

data/cc0.ply:
	@mkdir -p data
	./download_data.sh 1dohIKUrPq4PP9VSTSE47KndNpaETMbvw $@
	@rm -f cookie

data/dragon_curvs.ply:
	@mkdir -p data
	./download_data.sh 1q-JU_I6xBs4J-S016RlYBVleeujdlH64 $@
	@rm -f cookie

data/happy_buddha_curvs.ply:
	@mkdir -p data
	./download_data.sh 1NcSGA3uov1uzM9npSXFE8xBpgIfNxP-c $@
	@rm -f cookie

data/lucy_simple_curvs.ply:
	@mkdir -p data
	./download_data.sh 17-dkMQ89STR9feOq18xmSeHB0zmXkRu8 $@
	@rm -f cookie

results/armadillo_biased_curvatures_sdf/models/model_best.pth: data/armadillo_curvs.ply
	python main.py experiments/armadillo_curvature_batch_sdf.json

results/armadillo_biased_curvatures_sdf/models/model_best_biases.bin: results/armadillo_biased_curvatures_sdf/models/model_best.pth
	python Shader-Neural-Implicits/tools/weights_biases_from_pth.py -f=results/armadillo_biased_curvatures_sdf/models/model_best.pth
