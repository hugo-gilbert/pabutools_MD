"""
Print pooled head-to-head counts for L^1 against L^infty, one row per
satisfaction function.

Each cell is "w / l / t": the number of the 200 trials (10 datasets x 20
cost-splittings) on which L^1 scored higher, L^infty scored higher, and the two
tied. Both projections see the same splittings, so every trial is paired.
"""

import json
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
INF_ALPHA = 9999.0
TOL = 1e-9
SATS = ["cardinality", "max_cost", "sum_cost", "random_additive"]
BASE, TOP = "sum", "max"
ALGOS = {"mes": "MES", "phragmen": "Phragmen"}
DATASETS = [p.stem for p in sorted((DATA / "instances_all").glob("*.pb"))]


def cell(fam, key, sat, metric):
    base_wins = top_wins = ties = 0
    for stem in DATASETS:
        trials = json.load(open(DATA / f"{fam}_{sat}" / stem / "results.json"))[
            stem + ".pb"]["_trials"]
        for t in trials:
            if metric == "gamma":
                a, b = (0.0 if t[key][p]["alpha"] >= INF_ALPHA
                        else 1.0 / t[key][p]["alpha"] for p in (BASE, TOP))
            else:
                a, b = (float(t[key][p]["social_welfare"]) for p in (BASE, TOP))
            if abs(b - a) <= TOL * max(1.0, abs(a), abs(b)):
                ties += 1
            elif b > a:
                top_wins += 1
            else:
                base_wins += 1
    return f"{base_wins} / {top_wins} / {ties}"


pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

columns = {f"{key}·{mlbl}": [cell(fam, key, sat, metric) for sat in SATS]
           for fam, key in ALGOS.items()
           for metric, mlbl in [("gamma", "γ̂"), ("welfare", "SW")]}

print()
print('Pooled over 200 trials. Cells are "L1 wins / Linf wins / ties".')
print(pd.DataFrame(columns, index=SATS).to_string())
print()
