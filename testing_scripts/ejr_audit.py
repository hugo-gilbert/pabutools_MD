"""
Participatory Budgeting Testing Pipeline
=========================================
Reads every .pb file in a testing directory, constructs a multi-dimensional
election instance, runs MES with each configured projection function, audits
EJR for each, and writes structured logs.

To add a new projection function, append a (name, function) tuple to
PROJECTION_FUNCTIONS near the bottom of this file.
"""

import gurobipy as gp
from gurobipy import GRB
import sys
import os
import json
import random
import numpy as np
from datetime import datetime
from pathlib import Path

# --- Adjust this path as needed ---
sys.path.insert(0, '/Users/bencookson/Documents/Shah/PB/pabutools_MD')

from pabutools.rules.md_mes.md_mes_rule import *
from pabutools.election import *
from pabutools.election.satisfaction.md_satisfaction import *
from pabutools.election.satisfaction.satisfactionprofile import *
from pabutools.election.md_pabulib import md_parse_pabulib_random_split


# =====================================================================
# Helper utilities (unchanged from your original code)
# =====================================================================

def check_infinite_deviation(votes, costs, budget, current_utilities, sat_values):
    zero_util_voters = [i for i, u in enumerate(current_utilities) if u == 0]
    if not zero_util_voters:
        return False, [], []

    num_voters = len(votes)
    num_projects = len(costs)
    dimension = len(costs[0])

    m = gp.Model("Zero_Check")
    m.setParam('OutputFlag', 0)

    y = m.addVars(num_projects, vtype=GRB.BINARY, name="y")
    x = m.addVars(num_voters, vtype=GRB.BINARY, name="x")

    for i in range(num_voters):
        if i not in zero_util_voters:
            m.addConstr(x[i] == 0)

    for d in range(dimension):
        m.addConstr(
            (num_voters * gp.quicksum(costs[j][d] * y[j] for j in range(num_projects)))
            <= (budget * gp.quicksum(x[i] for i in range(num_voters)))
        )

    M_prime = 3
    for j in range(num_projects):
        for i in zero_util_voters:
            m.addConstr(votes[i][j] + M_prime * (2 - x[i] - y[j]) >= 1)

    m.addConstr(gp.quicksum(x[i] for i in zero_util_voters) >= 1)
    m.addConstr(gp.quicksum(y[j] * sat_values[j] for j in range(num_projects)) >= 1)

    m.setObjective(0, GRB.MINIMIZE)
    m.optimize()

    if m.Status == GRB.OPTIMAL:
        return True, [y[j].X for j in range(num_projects)], [x[i].X for i in range(num_voters)]
    return False, [], []


def calculate_ejr_approximation_approval(votes, costs, budget, outcome_set, sat_values=None):
    num_voters = len(votes)
    num_projects = len(costs)
    dimension = len(costs[0])

    if sat_values is None:
        sat_values = [1] * num_projects

    current_utilities = [
        sum(votes[i][j] * outcome_set[j] * sat_values[j] for j in range(num_projects))
        for i in range(num_voters)
    ]

    is_infinite, inf_T, inf_S = check_infinite_deviation(
        votes, costs, budget, current_utilities, sat_values
    )
    if is_infinite:
        return 9999.0, inf_T, inf_S

    max_sat_sum = sum(sat_values)
    M = max_sat_sum * 1000
    littleM = 0.001

    m = gp.Model("EJR_Approximation")
    m.setParam('OutputFlag', 0)
    m.setParam('TimeLimit', 60)

    y = m.addVars(num_projects, vtype=GRB.BINARY, name="y")
    x = m.addVars(num_voters, vtype=GRB.BINARY, name="x")
    alpha = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="alpha")

    for d in range(dimension):
        m.addConstr(
            (num_voters * gp.quicksum(costs[j][d] * y[j] for j in range(num_projects)))
            <= (budget * gp.quicksum(x[i] for i in range(num_voters)))
        )

    for j in range(num_projects):
        for i in range(num_voters):
            m.addConstr(votes[i][j] + M * (2 - x[i] - y[j]) >= 1)

    for i in range(num_voters):
        new_util = gp.quicksum(y[j] * sat_values[j] for j in range(num_projects))
        if current_utilities[i] > 0:
            m.addConstr(alpha * current_utilities[i] + littleM - M * (1 - x[i]) <= new_util)

    m.addConstr(gp.quicksum(x[i] for i in range(num_voters)) >= 1)
    m.addConstr(gp.quicksum(y[j] for j in range(num_projects)) >= 1)

    m.setObjective(alpha, GRB.MAXIMIZE)
    m.optimize()

    if m.Status == GRB.OPTIMAL:
        return alpha.X, [y[j].X for j in range(num_projects)], [x[i].X for i in range(num_voters)]
    elif m.Status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE):
        return 1.0, [], []
    else:
        print(f"Solver exited with unexpected status {m.Status}")
        return 1.0, [], []


def ballot_to_vote_list(ballot, project_list):
    return [1 if project_list[p] in ballot else 0 for p in range(len(project_list))]


def generate_approval_list_cost(project_list):
    util_map = {project: 1 for project in project_list}
    util_vector = [1] * len(project_list)
    return util_map, util_vector


def generate_cardinal_normal_list_cost(project_list, variance_param=0.25):
    util_map = {}
    util_vector = []
    for project in project_list:
        val = int(max(1, np.random.normal(sum(project.costs), variance_param * sum(project.costs))))
        util_vector.append(val)
        util_map[project] = val
    return util_map, util_vector


def generate_cardinal_uniform_list(project_list, low=1, high=100):
    """Assign each project a satisfaction value drawn uniformly from [low, high]."""
    util_map = {}
    util_vector = []
    for project in project_list:
        val = random.randint(low, high)
        util_vector.append(val)
        util_map[project] = val
    return util_map, util_vector


class RandomAdditiveSatGenerator:
    def __init__(self, project_values: dict):
        self.project_values = project_values

    def __call__(self, instance, profile, ballot, project, precomputed_values):
        if project in ballot:
            return self.project_values.get(project, 0)
        return 0


# =====================================================================
# Logging helpers
# =====================================================================

def write_instance_log(log_dir: Path, instance, profile, project_list, category_list, util_vector):
    """Write a human-readable log describing the election instance."""
    path = log_dir / "instance_log.txt"
    with open(path, "w") as f:
        f.write("=" * 72 + "\n")
        f.write("ELECTION INSTANCE LOG\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 72 + "\n\n")

        # Meta
        f.write("--- Meta ---\n")
        for k, v in instance.meta.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        # Dimension mapping
        f.write("--- Dimension → Category Mapping ---\n")
        for idx, cat in enumerate(category_list):
            f.write(f"  Dimension {idx}: {cat}\n")
        f.write(f"  Total dimensions: {len(category_list)}\n\n")

        # Budget
        f.write("--- Budget Limits (per dimension) ---\n")
        for idx, b in enumerate(instance.budget_limits):
            f.write(f"  Dimension {idx} ({category_list[idx]}): {b}\n")
        f.write("\n")

        # Projects
        f.write("--- Projects ---\n")
        f.write(f"  Total: {len(project_list)}\n\n")
        for j, project in enumerate(project_list):
            cats = instance.project_meta[project].get("categories", set())
            f.write(f"  [{j}] {project.name}\n")
            f.write(f"       Categories : {', '.join(sorted(cats))}\n")
            f.write(f"       Cost vector: {project.costs}\n")
            cost_pcts = []
            for d in range(len(category_list)):
                c = float(project.costs[d])
                b = float(instance.budget_limits[d])
                pct = (c / b * 100) if b > 0 else 0.0
                cost_pcts.append(f"{pct:.1f}%")
            f.write(f"       % of budget: [{', '.join(cost_pcts)}]\n")
            f.write(f"       Sat value  : {util_vector[j]}\n")
            f.write("\n")

        # Profile summary
        f.write("--- Voter Profile ---\n")
        f.write(f"  Number of ballots: {profile.num_ballots()}\n")
        f.write(f"  Vote type        : {instance.meta.get('vote_type', 'unknown')}\n")


def write_results_log(log_dir: Path, proj_name: str, instance, project_list, category_list,
                      outcome, v_outcome, approx, S, T, elapsed_seconds):
    """Write a human-readable log of the MES outcome and EJR audit for one projection function."""
    path = log_dir / f"{proj_name}_results_log.txt"
    with open(path, "w") as f:
        f.write("=" * 72 + "\n")
        f.write(f"RESULTS LOG  [{proj_name}]\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 72 + "\n\n")

        # Outcome
        selected = [project_list[j] for j in range(len(project_list)) if v_outcome[j] == 1]
        f.write("--- Selected Projects (MES Outcome) ---\n")
        f.write(f"  Count: {len(selected)} / {len(project_list)}\n\n")
        total_cost_vec = [0] * len(category_list)
        for p in selected:
            cats = instance.project_meta[p].get("categories", set())
            f.write(f"  • {p.name}\n")
            f.write(f"    Categories : {', '.join(sorted(cats))}\n")
            f.write(f"    Cost vector: {p.costs}\n")
            cost_pcts = []
            for d in range(len(category_list)):
                c = float(p.costs[d])
                b = float(instance.budget_limits[d])
                pct = (c / b * 100) if b > 0 else 0.0
                cost_pcts.append(f"{pct:.1f}%")
            f.write(f"    % of budget: [{', '.join(cost_pcts)}]\n")
            for d in range(len(category_list)):
                total_cost_vec[d] += p.costs[d]
        f.write("\n")

        f.write("--- Total Cost of Selected Bundle (per dimension) ---\n")
        for d, cat in enumerate(category_list):
            budget_d = float(instance.budget_limits[d])
            cost_d = float(total_cost_vec[d])
            pct = (cost_d / budget_d * 100) if budget_d > 0 else 0.0
            f.write(f"  Dim {d} ({cat}): {cost_d:.4f} / {budget_d:.4f}"
                    f"  ({pct:.1f}% used)\n")
        f.write("\n")

        # EJR
        f.write("--- EJR Audit ---\n")
        if approx >= 9000:
            f.write("  Result: INFINITE DEVIATION (alpha = 9999)\n")
            f.write("  A coalition of zero-utility voters can afford a positive-utility bundle.\n")
        else:
            f.write(f"  Alpha (max improvement factor): {approx:.6f}\n")
            if approx <= 1.0:
                f.write("  Interpretation: Outcome satisfies EJR (alpha <= 1).\n")
            else:
                f.write(f"  Interpretation: EJR violated by factor {approx:.4f}.\n")
        f.write("\n")

        if S:
            deviating_voters = [i for i, val in enumerate(S) if val > 0.5]
            f.write(f"  Deviating coalition (voter indices): {deviating_voters}\n")
        if T:
            deviating_projects = [j for j, val in enumerate(T) if val > 0.5]
            f.write(f"  Deviating bundle (project indices) : {deviating_projects}\n")
            f.write(f"  Deviating bundle (project names)   : "
                    f"{[project_list[j].name for j in deviating_projects]}\n")
        f.write("\n")

        f.write(f"  Wall-clock time: {elapsed_seconds:.2f}s\n")


# =====================================================================
# Main pipeline
# =====================================================================

def run_single_file(pb_path: str, output_root: Path, projection_functions: list,
                    seed: int | None = None):
    """
    Process one .pb file with all projection functions and write logs.

    projection_functions: list of (name: str, func: callable) tuples.
    Returns (file_name, {proj_name: alpha, ...}).
    """
    file_name = os.path.basename(pb_path)
    log_dir = output_root / file_name.replace(".pb", "")
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Processing: {file_name}")
    print(f"Log folder: {log_dir}")
    print(f"{'=' * 60}")

    # ---- 1. Parse ----
    instance, profile = md_parse_pabulib_random_split(pb_path, seed=seed)
    project_list = list(instance)

    from pabutools.election.md_pabulib import get_all_categories
    with open(pb_path, "r", encoding="utf-8-sig") as f:
        category_list = get_all_categories(f.read())

    dim = len(instance.budget_limits)
    budget = instance.budget_limits[0]
    personal_budget = budget / profile.num_ballots()

    # ---- 2. Generate utilities once (shared across all projection functions) ----
    util_map, util_vector = generate_cardinal_normal_list_cost(project_list)
    sat_func = RandomAdditiveSatGenerator(util_map)

    sat_measures_list = [
        AdditiveSatisfaction(
            instance=instance,
            profile=profile,
            ballot=ballot,
            func=sat_func,
        )
        for ballot in profile
    ]
    sat_profile = SatisfactionProfile(init=sat_measures_list, instance=instance)
    sat_profile.sat_class = AdditiveSatisfaction

    # ---- 3. Write shared instance log ----
    write_instance_log(log_dir, instance, profile, project_list, category_list, util_vector)
    print(f"  Instance log written ({len(project_list)} projects, {dim} dimensions, "
          f"{profile.num_ballots()} voters)")

    # Build shared EJR input vectors (same for all projection functions)
    v_votes = [ballot_to_vote_list(ballot, project_list) for ballot in profile]
    v_costs = [p.costs for p in project_list]

    # ---- 4. Run MES + EJR for each projection function ----
    results = {}
    for proj_name, proj_func in projection_functions:
        print(f"\n  --- Projection: {proj_name} ---")

        mes_log_path = str(log_dir / f"{proj_name}_mes_execution_log.txt")
        start = datetime.now()
        outcome = naive_md_mes(
            instance,
            profile,
            Cardinality_Sat,
            [personal_budget for _ in range(dim)],
            proj_func,
            sat_profile,
            log_path=mes_log_path,
        )
        elapsed_mes = (datetime.now() - start).total_seconds()
        print(f"    MES completed in {elapsed_mes:.2f}s  —  {len(outcome)} projects selected")
        print(f"    MES execution log: {mes_log_path}")

        v_outcome = [1 if p in outcome else 0 for p in project_list]

        start = datetime.now()
        approx, T, S = calculate_ejr_approximation_approval(
            v_votes, v_costs, budget, v_outcome, sat_values=util_vector
        )
        elapsed_ejr = (datetime.now() - start).total_seconds()
        if approx >= 9000:
            print(f"    EJR audit completed in {elapsed_ejr:.2f}s  —  INFINITE DEVIATION DETECTED")
        else:
            print(f"    EJR audit completed in {elapsed_ejr:.2f}s  —  alpha = {approx}")

        write_results_log(
            log_dir, proj_name, instance, project_list, category_list,
            outcome, v_outcome, approx, S, T, elapsed_mes + elapsed_ejr,
        )
        print(f"    Results log: {log_dir / f'{proj_name}_results_log.txt'}")

        results[proj_name] = approx

    return file_name, results


def run_pipeline(
    testing_dir: str = "testing_files",
    output_dir: str = "output_logs",
    projection_functions: list | None = None,
    seed: int | None = 42,
):
    """
    Discover all .pb files in testing_dir and process each one with all
    configured projection functions.
    """
    if projection_functions is None:
        projection_functions = PROJECTION_FUNCTIONS

    testing_path = Path(testing_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    pb_files = sorted(testing_path.glob("*.pb"))
    if not pb_files:
        print(f"No .pb files found in {testing_path.resolve()}")
        return

    proj_names = [name for name, _ in projection_functions]
    print(f"Found {len(pb_files)} .pb file(s) in {testing_path.resolve()}")
    print(f"Projection functions: {proj_names}")
    print(f"Output directory: {output_root.resolve()}\n")

    summary = []
    for pb_file in pb_files:
        try:
            name, alphas = run_single_file(
                str(pb_file), output_root, projection_functions, seed=seed
            )
            summary.append({"file": name, "alphas": alphas, "status": "ok"})
        except Exception as e:
            print(f"  ERROR processing {pb_file.name}: {e}")
            summary.append({"file": pb_file.name, "alphas": {}, "status": str(e)})

    # ---- Write summary across all files ----
    summary_path = output_root / "summary.txt"
    col_w = 14  # width per projection-function alpha column

    with open(summary_path, "w") as f:
        f.write("=" * 72 + "\n")
        f.write("PIPELINE SUMMARY\n")
        f.write(f"Run at: {datetime.now().isoformat()}\n")
        f.write(f"Files processed : {len(summary)}\n")
        f.write(f"Projection funcs: {', '.join(proj_names)}\n")
        f.write("=" * 72 + "\n\n")

        ok_runs = [s for s in summary if s["status"] == "ok"]
        failed_runs = [s for s in summary if s["status"] != "ok"]

        f.write(f"Successful: {len(ok_runs)}\n")
        f.write(f"Errors    : {len(failed_runs)}\n\n")

        # Per-projection stats
        for proj_name in proj_names:
            finite = [s["alphas"][proj_name] for s in ok_runs
                      if proj_name in s["alphas"] and s["alphas"][proj_name] < 9000]
            infinite = [s for s in ok_runs
                        if proj_name in s["alphas"] and s["alphas"][proj_name] >= 9000]
            f.write(f"[{proj_name}]\n")
            f.write(f"  Finite alpha  : {len(finite)}\n")
            f.write(f"  Infinite (EJR): {len(infinite)}\n")
            if finite:
                f.write(f"  Mean alpha    : {np.mean(finite):.6f}\n")
                f.write(f"  Min  alpha    : {np.min(finite):.6f}\n")
                f.write(f"  Max  alpha    : {np.max(finite):.6f}\n")
            f.write("\n")

        # Side-by-side table
        f.write("-" * (55 + col_w * len(proj_names) + 10) + "\n")
        header = f"{'File':<55}"
        for pn in proj_names:
            header += f"  {pn:>{col_w - 2}}"
        header += "  Status"
        f.write(header + "\n")
        f.write("-" * (55 + col_w * len(proj_names) + 10) + "\n")

        for s in summary:
            row = f"{s['file']:<55}"
            for pn in proj_names:
                alpha = s["alphas"].get(pn)
                if alpha is None:
                    astr = "N/A"
                elif alpha >= 9000:
                    astr = "INFINITE"
                else:
                    astr = f"{alpha:.4f}"
                row += f"  {astr:>{col_w - 2}}"
            row += f"  {s['status']}"
            f.write(row + "\n")

    print(f"\nPipeline complete. Summary written to {summary_path.resolve()}")


# =====================================================================
# Projection functions
# =====================================================================
# Add new entries here to include them in every pipeline run.
# Each entry is a (display_name, function) tuple.

PROJECTION_FUNCTIONS = [
    ("max", projection_function_max),
    ("sum", projection_function_sum),
]


if __name__ == "__main__":
    run_pipeline(
        testing_dir="testing_scripts/testing_files",
        output_dir="testing_scripts/output_logs",
        seed=42,
    )
