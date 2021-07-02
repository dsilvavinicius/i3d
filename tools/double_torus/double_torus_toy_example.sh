#!/bin/bash
echo "Preprocessing double torus..."
./tools/double_torus/preprocess_double_torus.sh

echo "Training double torus..."
./tools/double_torus/train_double_torus.sh

echo "Creating binary neural network representation..."
./tools/double_torus/double_torus_pth2bin.sh