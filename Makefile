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

.PHONY: all clean
