#!/bin/sh
# Run experiments from a jobs file, N at a time:  experiments/queue.sh 8 jobs.txt
# Each line of the jobs file:  <name> <seed> <steps> '<json overrides>'
# Single-threaded XLA gives the same speed per process for this small model, so N ~ number of cores.
cd "/bin/.." || exit 1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
mkdir -p experiments/results
xargs -P "" -L 1 sh -c 'uv run python experiments/run_experiment.py "/bin/zsh" "" "" "" >> experiments/results/log.txt 2>&1' < ""
