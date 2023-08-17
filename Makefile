.PHONY: all clean comparison_analytic comparison_ply_cuda comparison_ply train_release_models viz_armadillo

all: viz_armadillo

clean:
	@rm -Rf results/armadillo results/bunny results/buddha results/dragon results/lucy_simple
	@rm -Rf __pycache__

comparison_analytic:
	python experiment_scripts/comparison_analytic.py --training_points 5000 --test_points 5000 --input sphere --fraction_on_surface 0.5 --methods rbf siren i3d i3dcurv --mc_resolution 128 --num_runs 100
	python experiment_scripts/comparison_analytic.py --training_points 5000 --test_points 5000 --input torus --fraction_on_surface 0.5 --methods rbf siren i3d i3dcurv --mc_resolution 128 --num_runs 100

comparison_ply_cuda:
	python experiment_scripts/comparison_ply.py --methods rbf siren i3d --resolution 256 --num_runs 10 --device cuda

comparison_ply:
	python experiment_scripts/comparison_ply.py --methods rbf siren i3d --resolution 256 --num_runs 10 --device cpu

train_release_models: results/armadillo/best.ply results/bunny/best.ply results/dragon/best.ply results/buddha/best.ply results/lucy_simple/best.ply
	@echo "Models trained. Check the \"results\" folder"

results/armadillo/best.ply: results/armadillo/best.pth
	@python tools/reconstruct.py $< $@ -r 350

results/armadillo/best.pth: data/armadillo_curvs.ply
	@python train_sdf.py $< results/armadillo/ experiments/default_curvature.yaml --curvature-fractions 0.1 0.4 0.5

data/armadillo_curvs.ply:
	@mkdir -p data
	./download_data.sh 14y6MxrOPCLR2yggjjdYnvD9VAljhh024 $@
	@rm -f cookie

results/bunny/best.ply: results/bunny/best.pth
	@python tools/reconstruct.py $< $@ -r 350

results/bunny/best.pth: data/bunny_curvs.ply
	@python train_sdf.py $< results/bunny/ experiments/default_curvature.yaml --omega0 30 --omegaW 30

data/bunny_curvs.ply:
	@mkdir -p data
	./download_data.sh 1gKYPtQ2hvAdjjKsDEJPQvuczLOm29d9o $@
	@rm -f cookie

data/cc0.ply:
	@mkdir -p data
	./download_data.sh 1dohIKUrPq4PP9VSTSE47KndNpaETMbvw $@
	@rm -f cookie

results/dragon/best.ply: results/dragon/best.pth
	@python tools/reconstruct.py $< $@ -r 350

results/dragon/best.pth: data/dragon_curvs.ply
	@python train_sdf.py $< results/dragon/ experiments/default_curvature.yaml

data/dragon_curvs.ply:
	@mkdir -p data
	./download_data.sh 1q-JU_I6xBs4J-S016RlYBVleeujdlH64 $@
	@rm -f cookie

results/buddha/best.ply: results/buddha/best.pth
	@python tools/reconstruct.py $< $@ -r 350

results/buddha/best.pth: data/buddha_curvs.ply
	@python train_sdf.py $< results/buddha/ experiments/default_curvature.yaml

data/buddha_curvs.ply:
	@mkdir -p data
	./download_data.sh 1NcSGA3uov1uzM9npSXFE8xBpgIfNxP-c $@
	@rm -f cookie

results/lucy_simple/best.ply: results/lucy_simple/best.pth
	@python tools/reconstruct.py $< $@ -r 350

results/lucy_simple/best.pth: data/lucy_simple_curvs.ply
	@python train_sdf.py $< results/lucy_simple/ experiments/default_curvature.yaml

data/lucy_simple_curvs.ply:
	@mkdir -p data
	./download_data.sh 17-dkMQ89STR9feOq18xmSeHB0zmXkRu8 $@
	@rm -f cookie

viz_armadillo: results/armadillo/best.ply
	@meshlab $<
