"""
Print EJR-approximation and social welfare summary tables, one per satisfaction function.

Rows are every dataset in data/instances_all; columns are MD-MES and MD-Phragmen
across the 5 L^p projections. Each cell is the mean and 95% Student's-t
confidence interval of gamma-hat = 1/alpha (or social welfare) over that dataset's trials.
"""

import json
from pathlib import Path

import pandas as pd
from scipy import stats

DATA = Path(__file__).resolve().parent.parent / "data"
INF_ALPHA = 9999.0
SATS = ["cardinality", "max_cost", "sum_cost", "random_additive"]
PROJS = {"sum": "L1", "L2": "L2", "L3": "L3", "L4": "L4", "max": "Linf"}
ALGOS = {"mes": "MES", "phragmen": "Phragmen"}
# welfare display scale per satisfaction function: (divisor, unit label, decimals)
WELFARE_UNIT = {"cardinality": (1e3, "k", 2), "random_additive": (1e3, "k", 1),
                "max_cost": (1e6, "M", 1), "sum_cost": (1e6, "M", 1)}
DATASETS = [p.stem for p in sorted((DATA / "instances_all").glob("*.pb"))]


def cell(fam, key, sat, stem, proj):
    f = DATA / f"{fam}_{sat}" / stem / "results.json"
    if not f.exists():
        return "n/a"
    trials = json.load(open(f))[stem + ".pb"]["_trials"]
    vals = [0.0 if t[key][proj]["alpha"] >= INF_ALPHA else 1.0 / t[key][proj]["alpha"]
            for t in trials]
    mean = sum(vals) / len(vals)
    sem = stats.sem(vals) if len(vals) > 1 else 0.0
    hw = stats.t.interval(0.95, len(vals) - 1, mean, sem)[1] - mean if sem else 0.0
    return f"{mean:.2f}±{hw:.2f}"

def cell_welfare(fam, key, sat, stem, proj):
    f = DATA / f"{fam}_{sat}" / stem / "results.json"
    if not f.exists():
        return "n/a"
    trials = json.load(open(f))[stem + ".pb"]["_trials"]
    vals = [t[key][proj]["social_welfare"] for t in trials]
    mean = sum(vals) / len(vals)
    sem = stats.sem(vals) if len(vals) > 1 else 0.0
    hw = stats.t.interval(0.95, len(vals) - 1, mean, sem)[1] - mean if sem else 0.0
    div, _, dec = WELFARE_UNIT[sat]
    return f"{mean/div:.{dec}f}±{hw/div:.{dec}f}"


pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

print()
for sat in SATS:
    columns = {f"{key}·{plbl}": [cell(fam, key, sat, stem, proj) for stem in DATASETS]
               for fam, key in ALGOS.items() for proj, plbl in PROJS.items()}
    df = pd.DataFrame(columns, index=[s.replace("Poland_Warszawa_", "").replace("poland_warszawa_", "")
                                      for s in DATASETS])
    print(f"\n### {sat} — EJR γ̂ = 1/α (mean ± 95% t-CI) ###")
    print(df.to_string())

for sat in SATS:
    columns = {f"{key}·{plbl}": [cell_welfare(fam, key, sat, stem, proj) for stem in DATASETS]
               for fam, key in ALGOS.items() for proj, plbl in PROJS.items()}
    df = pd.DataFrame(columns, index=[s.replace("Poland_Warszawa_", "").replace("poland_warszawa_", "")
                                      for s in DATASETS])
    print(f"\n### {sat} — Social Welfare in {WELFARE_UNIT[sat][1]} (mean ± 95% t-CI) ###")
    print(df.to_string())
