# Experiment scripts

The tools behind [docs/experiments.md](../docs/experiments.md). They train without run folders or a
dashboard, so many can run side by side, and they append their results to `results/`.

| Script | What it does |
|---|---|
| `run_experiment.py <name> <seed> <steps> '<json>'` | Train once with config overrides (`{"env": {...}, "ppo": {...}, "cur": {...}}`), evaluate, append a line to `results/results.jsonl`, save the weights to `results/params/` |
| `queue.sh <N> <jobs.txt>` | Run the lines of a jobs file (`name seed steps 'json'`), N at a time, single-threaded XLA |
| `robust_eval.py <run dir or .pkl>` | The stricter test: hanging still / perturbed, random starts, pushes, settle times, jerk |
| `robust_table.py` | `robust_eval` of all saved policies on the same episodes → `results/robust_table.json` |
| `pbt.py <name> <pop> <steps> <interval> '<json>'` | Population-based training (exploit + greedy weight soup) → `results/pbt.jsonl` |
| `compare_runs.py runs/<group>/*` | Final checkpoints of training runs on the same 256 episodes per case, with and without the mirror rule → `results/compare_<group>.json` |
| `summarize.py [prefix ...]` | Per-experiment summary across seeds |
| `import_runs.py <prefix ...>` | Copy saved weights into `runs/imported/` so the dashboard can show them |
| `plot_figures.py` | The figures in `docs/img/` (`uv run --with matplotlib experiments/plot_figures.py`) |

Example, two configurations with 4 seeds each:

```bash
cat > jobs.txt <<'EOF'
base 1 60e6 '{}'
base 2 60e6 '{}'
noease 1 60e6 '{"cur": {"gravity_relief": 0.0, "damping": 0.0}}'
noease 2 60e6 '{"cur": {"gravity_relief": 0.0, "damping": 0.0}}'
EOF
experiments/queue.sh 8 jobs.txt
uv run experiments/summarize.py base noease
```

## results/

- `results.jsonl`: every experiment run during the project, one line per seed, with its overrides
  (`ov`), the evaluator result, off-track statistics, the final curriculum state and, from round
  3 on, the `robust` evaluation.
  - Rows before the `H_` experiments come from earlier versions of the code (50 Hz, a different
    start distribution and curriculum), so their overrides don't all apply to the current code. They
    are kept as a record. The experiment names map to the rounds in docs/experiments.md:
    `base`/`tb*`/`es*`/`up*`/`L_*`/`X_*`/`N_*` = round 1, `H_*` = round 2, `G_*`/`B_3L`/`EN` = round 3.
- `pbt.jsonl`: the population run, with member scores at every exploit step.
- `robust_table.json`: the data behind `docs/img/robustness.png`.
- `params/`: saved weights. Not in git (model files are excluded from the repo); regenerate them by
  re-running the experiments.
