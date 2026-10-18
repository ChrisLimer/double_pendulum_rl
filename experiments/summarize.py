"""Summarize experiments/results/results.jsonl by experiment name, one line per metric across seeds.

    uv run experiments/summarize.py EN G_g05     # names starting with these (all if none given)
"""
import collections
import json
import sys
from pathlib import Path

PATH = Path(__file__).resolve().parent / "results" / "results.jsonl"
g = collections.defaultdict(list)
for l in open(PATH):
    r = json.loads(l)
    if any(r['name'].startswith(p) for p in (sys.argv[1:] or [''])): g[r['name']].append(r)
for k, v in g.items():
    v.sort(key=lambda r: r['seed'])
    print(f"== {k}  seeds {[r['seed'] for r in v]}")
    print(f"   eval success {[round(r['success'],2) for r in v]}  cur {[tuple(round(x,2) for x in r['final_cur']) for r in v]}")
    if 'robust' in v[0]:
        for case in v[0]['robust']:
            print(f"   {case:17s} succ {[r['robust'][case]['success'] for r in v]}  settled_med {[r['robust'][case]['t_settled_med'] for r in v]}  off {[r['robust'][case]['off'] for r in v]}")
