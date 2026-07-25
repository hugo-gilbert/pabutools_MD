"""
MD-Phragmen pipeline.

Parses a PaBuLib election into a random multi-dimensional instance, runs
MD-Phragmen for every L^p projection function (sum, L2, L3, L4, max), audits each
outcome for EJR, and saves the results as JSON.

"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import gurobipy as gp
from gurobipy import GRB

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rules.md_mes.md_mes_rule import (
    make_lp_projection,
    projection_function_max, projection_function_sum, projection_function_l2,
)
from rules.md_phragmen.md_phragmen_continue_rule import projected_md_phragmen
from pabutools.election import Cardinality_Sat
from election.satisfaction.md_satisfaction import Max_Cost_Sat, Sum_Cost_Sat
from pabutools.election.satisfaction.additivesatisfaction import AdditiveSatisfaction
from election.md_pabulib import md_parse_pabulib_random_split, get_all_categories

SAT_CLASSES = {
    "cardinality":     (Cardinality_Sat, lambda p: 1),
    "max_cost":        (Max_Cost_Sat,    lambda p: max(p.costs)),
    "sum_cost":        (Sum_Cost_Sat,    lambda p: sum(p.costs)),
    "random_additive": (None, None),
}

PROJECTION_FUNCTIONS = [
    ("sum", projection_function_sum),
    ("L2",  projection_function_l2),
    ("L3",  make_lp_projection(3)),
    ("L4",  make_lp_projection(4)),
    ("max", projection_function_max),
]

INF_ALPHA = 9999.0


def make_random_sat_class(project_list, seed):
    """Fresh per-trial satisfaction class: each project gets a random integer
    utility in [1, 20], seeded deterministically from the trial seed."""
    rng = np.random.default_rng(seed + 100_000)
    values = {p: int(rng.integers(1, 21)) for p in project_list}

    def _sat_func(instance, profile, ballot, project, precomputed_values):
        return int(project in ballot) * values.get(project, 1)

    class _RandomAdditiveSat(AdditiveSatisfaction):
        def __init__(self, instance, profile, ballot):
            AdditiveSatisfaction.__init__(self, instance, profile, ballot, _sat_func)

    return _RandomAdditiveSat, values


def _ballot_to_vec(ballot, project_list):
    return [1 if project_list[p] in ballot else 0 for p in range(len(project_list))]


def _json_convert(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# =====================================================================
# EJR audit (Gurobi ILP)
# =====================================================================

def _check_infinite_deviation(votes, costs, budget, current_utilities, sat_values):
    zero_util = [i for i, u in enumerate(current_utilities) if u == 0]
    if not zero_util:
        return False, [], []

    nv, np_, nd = len(votes), len(costs), len(costs[0])
    cost_frac = [[costs[j][d] / budget for d in range(nd)] for j in range(np_)]

    m = gp.Model("Zero_Check")
    m.setParam('OutputFlag', 0)
    y = m.addVars(np_, vtype=GRB.BINARY, name="y")
    x = m.addVars(nv, vtype=GRB.BINARY, name="x")

    for i in range(nv):
        if i not in zero_util:
            m.addConstr(x[i] == 0)
    for d in range(nd):
        m.addConstr(nv * gp.quicksum(cost_frac[j][d] * y[j] for j in range(np_))
                    <= gp.quicksum(x[i] for i in range(nv)))
    for j in range(np_):
        for i in zero_util:
            m.addConstr(votes[i][j] + (2 - x[i] - y[j]) >= 1)
    m.addConstr(gp.quicksum(x[i] for i in zero_util) >= 1)
    m.addConstr(gp.quicksum(y[j] for j in range(np_)) >= 1)
    m.setObjective(0, GRB.MINIMIZE)
    m.optimize()

    if m.Status == GRB.OPTIMAL:
        return True, [y[j].X for j in range(np_)], [x[i].X for i in range(nv)]
    if m.Status == GRB.INFEASIBLE:
        return False, [], []
    raise RuntimeError(f"Zero-check terminated with status {m.Status}")


def _ejr_approximation(votes, costs, budget, outcome_set, sat_values):
    nv, np_, nd = len(votes), len(costs), len(costs[0])

    sat_scale = max(sat_values) if max(sat_values) > 0 else 1
    sat_norm  = [s / sat_scale for s in sat_values]
    cost_frac = [[costs[j][d] / budget for d in range(nd)] for j in range(np_)]

    current_util = [
        sum(votes[i][j] * outcome_set[j] * sat_norm[j] for j in range(np_))
        for i in range(nv)
    ]

    is_inf, inf_T, inf_S = _check_infinite_deviation(votes, costs, budget, current_util, sat_norm)
    if is_inf:
        return INF_ALPHA, inf_T, inf_S

    m = gp.Model("EJR")
    m.setParam('OutputFlag', 0)
    y     = m.addVars(np_, vtype=GRB.BINARY,      name="y")
    x     = m.addVars(nv,  vtype=GRB.BINARY,      name="x")
    alpha = m.addVar(lb=0.0, ub=10, vtype=GRB.CONTINUOUS, name="alpha")

    for d in range(nd):
        m.addConstr(nv * gp.quicksum(cost_frac[j][d] * y[j] for j in range(np_))
                    <= gp.quicksum(x[i] for i in range(nv)))
    for j in range(np_):
        for i in range(nv):
            m.addConstr(votes[i][j] + (2 - x[i] - y[j]) >= 1)
    for i in range(nv):
        new_u = gp.quicksum(y[j] * sat_norm[j] for j in range(np_))
        if current_util[i] > 0:
            m.addConstr((x[i] == 1) >> (alpha * current_util[i] <= new_u))

    m.addConstr(gp.quicksum(x[i] for i in range(nv)) >= 1)
    m.addConstr(gp.quicksum(y[j] for j in range(np_)) >= 1)
    m.setObjective(alpha, GRB.MAXIMIZE)
    m.optimize()

    if m.Status == GRB.OPTIMAL:
        return alpha.X, [y[j].X for j in range(np_)], [x[i].X for i in range(nv)]
    raise RuntimeError(f"EJR model terminated with status {m.Status}")


# =====================================================================
# Core pipeline
# =====================================================================

def run(pb_path: Path, sat_name: str, n_trials: int, output_path: Path,
        start_seed: int = 0, append: bool = False, budget_multiplier: float = 1.0):
    sat_class, sat_value_func = SAT_CLASSES[sat_name]

    with open(pb_path, "r", encoding="utf-8-sig") as fh:
        category_list = get_all_categories(fh.read())

    meta = None
    trials = []

    if append and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        entry = existing.get(pb_path.name, {})
        meta = entry.get("_meta")
        trials = entry.get("_trials", [])
        print(f"Appending to {output_path} ({len(trials)} existing trial(s))")

    end_seed = start_seed + n_trials - 1
    for seed in range(start_seed, start_seed + n_trials):
        print(f"-- seed {seed} (range {start_seed}..{end_seed})")
        instance, profile = md_parse_pabulib_random_split(
            str(pb_path), seed=seed, budget_multiplier=budget_multiplier)
        project_list = sorted(instance)

        if meta is None:
            meta = {
                "n_projects":    len(project_list),
                "n_voters":      profile.num_ballots(),
                "n_dims":        len(instance.budget_limits),
                "categories":    category_list,
                "budget_limits": [float(b) for b in instance.budget_limits],
                "vote_type":     instance.meta.get("vote_type", "unknown"),
                "project_names": [p.name for p in project_list],
            }

        dim          = len(instance.budget_limits)
        budget       = float(instance.budget_limits[0])
        personal_bgt = budget / profile.num_ballots()
        v_votes      = [_ballot_to_vec(b, project_list) for b in profile]
        v_costs      = [list(p.costs) for p in project_list]

        if sat_name == "random_additive":
            trial_sat_class, rand_values = make_random_sat_class(project_list, seed)
            sat_values = [rand_values[p] for p in project_list]
        else:
            trial_sat_class = sat_class
            sat_values = [sat_value_func(p) for p in project_list]
            rand_values = None

        sat_value_by_name = {p.name: v for p, v in zip(project_list, sat_values)}
        approval_by_name = {p.name: profile.approval_score(p) for p in project_list}

        sat_profile = profile.as_sat_profile(trial_sat_class)

        trial_data = {"seed": seed}
        if rand_values is not None:
            trial_data["sat_values"] = {p.name: v for p, v in rand_values.items()}

        proj_results = {}
        ejr_cache = {} 
        for proj_name, proj_func in PROJECTION_FUNCTIONS:
            outcome = projected_md_phragmen(
                instance, profile, trial_sat_class,
                [personal_bgt] * dim, proj_func, sat_profile,
            )
            v_outcome = [1 if p in outcome else 0 for p in project_list]
            selected  = [project_list[j] for j, v in enumerate(v_outcome) if v == 1]

            key = tuple(v_outcome)
            if key in ejr_cache:
                alpha, _, _ = ejr_cache[key]
            else:
                result = _ejr_approximation(v_votes, v_costs, budget, v_outcome, sat_values)
                ejr_cache[key] = result
                alpha, _, _ = result

            social_welfare = float(sum(sat_value_by_name[p.name] * approval_by_name[p.name]
                                       for p in selected))

            proj_results[proj_name] = {
                "alpha":          alpha,
                "n_selected":     len(selected),
                "selected_names": [p.name for p in selected],
                "social_welfare": social_welfare,
            }
            label = "INF" if alpha >= INF_ALPHA else f"alpha={alpha:.3f}"
            print(f"   [{proj_name}] {label}")

        trial_data["Phragmen"] = proj_results
        trials = [t for t in trials if t.get("seed") != seed]
        trials.append(trial_data)
        trials.sort(key=lambda t: t.get("seed", 0))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({pb_path.name: {"_meta": meta, "_trials": trials}},
                       f, indent=2, default=_json_convert)

    print(f"Done. Saved {len(trials)} trial(s) to {output_path}")


def _fmt_multiplier(c: float) -> str:
    """Format the budget multiplier for a path component: 2.0 -> '2', 1.5 -> '1.5'."""
    return str(int(c)) if float(c).is_integer() else str(c)


# Per-dataset base seed for the "instances_all" datasets: each gets a disjoint
# block of seeds (multiples of 1000) so no two datasets ever share a seed. When
# such a dataset is run, seeds start at its base instead of 0, and --start-seed
# is applied as an offset on top of the base (so --append extends within the
# block). Datasets not listed here use a base of 0.
DATASET_START_SEED = {
    "Poland_Warszawa_2018_Brodno": 0,
    "Poland_Warszawa_2019_Ochota": 1000,
    "Poland_Warszawa_2019_Praga-Polnoc": 2000,
    "Poland_Warszawa_2019_Rejon_Polnocno-Wschodni": 3000,
    "Poland_Warszawa_2019_Sielce": 4000,
    "Poland_Warszawa_2019_subunit_Wlochy": 5000,
    "Poland_Warszawa_2020_Rembertow": 6000,
    "poland_warszawa_2018_chrzanow-jelonki-polnocne-jelonki-poludniowe": 7000,
    "poland_warszawa_2018_rakowiec": 8000,
    "poland_warszawa_2018_saska-kepa": 9000,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pb_file", type=Path)
    parser.add_argument("--sat-class", default="cardinality", choices=list(SAT_CLASSES.keys()))
    parser.add_argument("--n-trials", type=int, default=30,
                         help="Number of seeds to run (default: 30). Runs seeds "
                              "start_seed through start_seed + n_trials - 1.")
    parser.add_argument("--start-seed", type=int, default=0,
                         help="First seed to run (default: 0).")
    parser.add_argument("--append", action="store_true",
                         help="Append to (and update by seed) an existing "
                              "results.json instead of overwriting it.")
    parser.add_argument("--budget-multiplier", type=float, default=1.0,
                         help="Scale the original budget B by this factor before "
                              "splitting across dimensions: total = c*B, per-dim = "
                              "c*B/d (default: 1.0). When != 1, the output dir is "
                              "suffixed with '_<c>x' so scaled runs don't overwrite "
                              "the c=1 results.")
    args = parser.parse_args()

    if not args.pb_file.exists():
        print(f"File not found: {args.pb_file}")
        sys.exit(1)

    suffix = f"_{args.sat_class}"
    if args.budget_multiplier != 1.0:
        suffix += f"_{_fmt_multiplier(args.budget_multiplier)}x"
    output_path = REPO_ROOT / "data" / f"phragmen{suffix}" / args.pb_file.stem / "results.json"

    run(args.pb_file, args.sat_class, args.n_trials, output_path,
        start_seed=DATASET_START_SEED.get(args.pb_file.stem, 0) + args.start_seed,
        append=args.append,
        budget_multiplier=args.budget_multiplier)


if __name__ == "__main__":
    main()
