#!/bin/bash

for i in {1..10}; do
    echo "Run ${i}"
    basename=run_${i}_armadillo_100epochs_curvatures2
    python -m cProfile -o ${basename}.prof experiment_scripts/sdf_for_n_iters.py > ${basename}.log
    gprof2dot -f pstats ${basename}.prof | dot -Tpng -o ${basename}.png
done
