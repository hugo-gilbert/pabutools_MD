"""
Multi-dimensional method of equal shares (with optional file logging).
"""

from __future__ import annotations

from copy import copy, deepcopy
from collections.abc import Iterable, Callable
from datetime import datetime

from pabutools.rules.budgetallocation import BudgetAllocation
from pabutools.rules.mes.mes_details import (
    MESAllocationDetails,
    MESIteration,
    MESProjectDetails,
)
from pabutools.utils import Numeric

from pabutools.election import AbstractApprovalProfile
from pabutools.election.satisfaction.satisfactionmeasure import GroupSatisfactionMeasure
from pabutools.election.ballot.ballot import AbstractBallot
from pabutools.election.md_instance import MDInstance, MDProject
from pabutools.election.profile import AbstractProfile
from pabutools.election.satisfaction import SatisfactionMeasure
from pabutools.tiebreaking import TieBreakingRule, lexico_tie_breaking
from pabutools.fractions import frac

import logging
logger = logging.getLogger(__name__)


class MDMESVoter:
    def __init__(self, index, ballot, sat, budget, multiplicity):
        self.index = index
        self.ballot = ballot
        self.sat = sat
        self.budget = budget
        self.multiplicity = multiplicity
        self.budget_over_sat_map = dict()

    def total_sat_project(self, proj):
        return self.multiplicity * self.sat.sat_project(proj)

    def total_budget(self):
        return [self.multiplicity * b for b in self.budget]

    def budget_over_sat_project(self, proj):
        res = self.budget_over_sat_map.get((proj, self.budget), None)
        if res is None:
            res = [frac(b, self.sat.sat_project(proj)) for b in self.budget]
            self.budget_over_sat_map[(proj, self.budget)] = res
        return res

    def __str__(self):
        return f"MESVoter[{self.budget}]"

    def __repr__(self):
        return f"MESVoter[{self.budget}]"


class MDMESProject(MDProject):
    def __init__(self, project):
        MDProject.__init__(self, project.name, project.dimension, project.costs)
        self.project = project
        self.total_sat = None
        self.sat_supporter_map = dict()
        self.unique_sat_supporter = None
        self.supporter_indices = []
        self.initial_affordability = None
        self.affordability = None

    def supporters_sat(self, supporter):
        if self.unique_sat_supporter:
            return self.unique_sat_supporter
        return supporter.sat.sat_project(self)

    def __str__(self):
        return f"MESProject[{self.name}, {float(self.affordability) if self.affordability is not None else '?'}]"

    def __repr__(self):
        return f"MESProject[{self.name}, {float(self.affordability) if self.affordability is not None else '?'}]"


def affordability_poor_rich_vector(voters, project):
    affordability_vector = []
    for dim in range(project.dimension):
        rich = set(project.supporter_indices)
        poor = set()
        while len(rich) > 0:
            poor_budget = sum(voters[i].total_budget()[dim] for i in poor)
            numerator = frac(project.costs[dim] - poor_budget)
            denominator = sum(voters[i].total_sat_project(project) for i in rich)
            affordability = frac(numerator, denominator)
            new_poor = {
                i for i in rich
                if voters[i].total_budget()[dim] < affordability * voters[i].sat.sat_project(project)
            }
            if len(new_poor) == 0:
                break
            rich -= new_poor
            poor.update(new_poor)
        affordability_vector.append(affordability)
    return affordability_vector


def projection_function_max(affordability_vector):
    return max(affordability_vector)

def projection_function_sum(affordability_vector):
    return sum(affordability_vector)

def projection_function_l2(affordability_vector):
    return sum(x**2 for x in affordability_vector)**(1/2)


# =====================================================================
# Log writer helper
# =====================================================================

class MESLogWriter:
    """Writes detailed MES iteration logs to a file."""

    def __init__(self, path):
        self.f = open(path, "w")

    def close(self):
        self.f.close()

    def _w(self, text=""):
        self.f.write(text + "\n")

    def header(self, instance, voters, projects, initial_budget_per_voter):
        dim = len(initial_budget_per_voter)
        self._w("=" * 80)
        self._w("MES DETAILED EXECUTION LOG")
        self._w(f"Generated: {datetime.now().isoformat()}")
        self._w("=" * 80)

        self._w()
        self._w(f"Dimensions          : {dim}")
        self._w(f"Number of voters    : {len(voters)}")
        self._w(f"Number of projects  : {len(projects)}")
        self._w(f"Initial voter budget: [{', '.join(f'{float(b):.4f}' for b in initial_budget_per_voter)}]")

        # --- Per-project summary ---
        self._w()
        self._w("-" * 80)
        self._w("PROJECT SUMMARY")
        self._w("-" * 80)
        for p in sorted(projects, key=lambda p: p.name):
            costs_str = ", ".join(f"{float(c):.4f}" for c in p.costs)
            supporter_count = len(p.supporter_indices)
            self._w(f"  {p.name}")
            self._w(f"    Costs       : [{costs_str}]")
            self._w(f"    Supporters  : {supporter_count}  (indices: {sorted(p.supporter_indices)})")
            self._w(f"    Total sat   : {float(p.total_sat):.4f}")
            # Per-supporter satisfaction breakdown
            sat_entries = []
            for i in sorted(p.supporter_indices):
                s = voters[i].sat.sat_project(p)
                sat_entries.append(f"v{i}={float(s):.2f}")
            self._w(f"    Sat detail  : {', '.join(sat_entries)}")
            self._w()

        # --- Per-voter summary ---
        self._w("-" * 80)
        self._w("VOTER SUMMARY")
        self._w("-" * 80)
        for v in voters:
            budget_str = ", ".join(f"{float(b):.4f}" for b in v.budget)
            approved = []
            for p in sorted(projects, key=lambda p: p.name):
                if v.index in p.supporter_indices:
                    approved.append(p.name)
            self._w(f"  Voter {v.index}")
            self._w(f"    Budget      : [{budget_str}]")
            self._w(f"    Multiplicity: {v.multiplicity}")
            self._w(f"    Approves    : {approved}")
            self._w()

    def iteration_start(self, iteration, remaining_count):
        self._w()
        self._w("=" * 80)
        self._w(f"ITERATION {iteration}")
        self._w(f"Remaining projects: {remaining_count}")
        self._w("=" * 80)

    def removed_projects(self, to_remove, voters, instance_dimension):
        if not to_remove:
            self._w("  No projects removed for being unaffordable.")
            return
        self._w()
        self._w(f"  --- Removed {len(to_remove)} unaffordable project(s) ---")
        for p in sorted(to_remove, key=lambda p: p.name):
            self._w(f"  {p.name}:")
            for dim in range(instance_dimension):
                supporter_budget = sum(float(voters[i].total_budget()[dim]) for i in p.supporter_indices)
                cost_d = float(p.costs[dim])
                shortfall = cost_d - supporter_budget
                self._w(f"    Dim {dim}: supporter budget = {supporter_budget:.4f}, "
                        f"cost = {cost_d:.4f}, shortfall = {shortfall:.4f}")

    def affordabilities(self, remaining_projects, affordabilities_vectors,
                        affordabilities_projections):
        self._w()
        self._w("  --- Affordability Table ---")
        self._w(f"  {'Project':<40} {'Vector':<40} {'Projection':>12}")
        self._w("  " + "-" * 92)

        ranked = sorted(remaining_projects,
                        key=lambda p: affordabilities_projections.get(p, float('inf')))
        for p in ranked:
            vec = affordabilities_vectors.get(p)
            proj_val = affordabilities_projections.get(p)
            if vec is not None:
                vec_str = "[" + ", ".join(f"{float(v):.6f}" for v in vec) + "]"
                proj_str = f"{float(proj_val):.6f}"
            else:
                vec_str = "N/A"
                proj_str = "N/A"
            self._w(f"  {p.name:<40} {vec_str:<40} {proj_str:>12}")

    def selection(self, selected, afford_vec, afford_proj):
        self._w()
        self._w(f"  >>> SELECTED: {selected.name}")
        vec_str = "[" + ", ".join(f"{float(v):.6f}" for v in afford_vec) + "]"
        self._w(f"      Affordability vector    : {vec_str}")
        self._w(f"      Affordability projection: {float(afford_proj):.6f}")
        costs_str = "[" + ", ".join(f"{float(c):.4f}" for c in selected.costs) + "]"
        self._w(f"      Cost vector             : {costs_str}")

    def budget_updates(self, selected, voters, afford_vec, instance_dimension):
        self._w()
        self._w("  --- Budget Updates for Supporters ---")
        self._w(f"  {'Voter':<8} {'Sat':>6} {'Dim':>4} {'Before':>12} {'Payment':>12} {'After':>12}")
        self._w("  " + "-" * 60)
        for i in sorted(selected.supporter_indices):
            v = voters[i]
            sat_val = float(v.sat.sat_project(selected))
            for dim in range(instance_dimension):
                before = float(v.budget[dim])
                payment = float(min(v.budget[dim], afford_vec[dim] * v.sat.sat_project(selected)))
                after = before - payment
                marker = " *DEPLETED*" if after < 1e-9 else ""
                self._w(f"  v{i:<6} {sat_val:>6.2f} {dim:>4} {before:>12.4f} {payment:>12.4f} {after:>12.4f}{marker}")

    def budget_snapshot(self, voters, instance_dimension):
        self._w()
        self._w("  --- Post-Iteration Budget Snapshot ---")
        for v in voters:
            budget_str = ", ".join(f"{float(b):.4f}" for b in v.budget)
            total = sum(float(b) for b in v.budget)
            self._w(f"  Voter {v.index}: [{budget_str}]  (total: {total:.4f})")

    def final_result(self, res):
        self._w()
        self._w("=" * 80)
        self._w("FINAL RESULT")
        self._w("=" * 80)
        self._w(f"  Selected {len(res)} project(s):")
        for i, p in enumerate(res):
            self._w(f"    {i+1}. {p.name}")
        self._w()
        self._w("END OF LOG")


# =====================================================================
# Main algorithm
# =====================================================================

def naive_md_mes(
    instance: MDInstance,
    profile: AbstractProfile,
    sat_class: type[SatisfactionMeasure],
    initial_budget_per_voter: list[Numeric],
    projection_function: Callable,
    sat_profile=None,
    log_path: str | None = None,
) -> BudgetAllocation:
    """
    Naive implementation of the method of equal shares with optional detailed logging.

    Parameters
    ----------
        instance: MDInstance
        profile: AbstractProfile
        sat_class: type[SatisfactionMeasure]
        initial_budget_per_voter: list[Numeric]
        projection_function: Callable
        sat_profile: optional pre-built satisfaction profile
        log_path: optional path to write a detailed iteration log
    """
    if sat_profile is None:
        sat_profile = profile.as_sat_profile(sat_class)

    log = MESLogWriter(log_path) if log_path else None

    voters = []
    instance_dimension = next(iter(instance)).dimension

    for index, sat in enumerate(sat_profile):
        voters.append(
            MDMESVoter(
                index, sat.ballot, sat,
                deepcopy(initial_budget_per_voter),
                sat_profile.multiplicity(sat),
            )
        )

    projects = set()
    for p in instance:
        mes_p = MDMESProject(p)
        total_sat = 0
        for i, v in enumerate(voters):
            indiv_sat = v.sat.sat_project(p)
            if indiv_sat > 0:
                total_sat += v.total_sat_project(p)
                mes_p.supporter_indices.append(i)
                mes_p.sat_supporter_map[v] = indiv_sat
        if total_sat > 0 and sum(p.costs) > 0:
            mes_p.total_sat = total_sat
            projects.add(mes_p)

    if log:
        log.header(instance, voters, projects, initial_budget_per_voter)

    res = BudgetAllocation()
    affordabilities_projections = dict()
    affordabilities_vectors = dict()

    remaining_projects = deepcopy(projects)
    iteration = 0
    while True:
        iteration += 1
        if log:
            log.iteration_start(iteration, len(remaining_projects))

        print(len(remaining_projects))

        to_remove = set()
        for project in remaining_projects:
            if any(
                sum(voters[i].total_budget()[dim] for i in project.supporter_indices) < project.costs[dim]
                for dim in range(instance_dimension)
            ):
                to_remove.add(project)
            else:
                afford_vector = affordability_poor_rich_vector(voters, project)
                if afford_vector is not None:
                    affordabilities_vectors[project] = afford_vector
                    affordabilities_projections[project] = projection_function(afford_vector)

        if log:
            log.removed_projects(to_remove, voters, instance_dimension)

        for project in to_remove:
            remaining_projects.remove(project)
            affordabilities_vectors.pop(project, None)
            affordabilities_projections.pop(project, None)

        if log:
            log.affordabilities(remaining_projects, affordabilities_vectors,
                                affordabilities_projections)

        if len(remaining_projects) == 0:
            if log:
                log.final_result(res)
                log.close()
            return res

        min_afford = min(affordabilities_projections.values())
        selected = [p for p in remaining_projects if affordabilities_projections[p] == min_afford][0]

        if log:
            log.selection(selected, affordabilities_vectors[selected], min_afford)
            log.budget_updates(selected, voters, affordabilities_vectors[selected], instance_dimension)

        res.append(selected.project)
        remaining_projects.remove(selected)

        for i in selected.supporter_indices:
            for dim in range(instance_dimension):
                voters[i].budget[dim] -= min(
                    voters[i].budget[dim],
                    affordabilities_vectors[selected][dim] * voters[i].sat.sat_project(selected)
                )

        del affordabilities_vectors[selected]
        del affordabilities_projections[selected]

        if log:
            log.budget_snapshot(voters, instance_dimension)