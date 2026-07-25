"""
MD-MES with greedy completion rule.

Runs the standard MD-MES algorithm, then fills any remaining budget capacity
with a greedy pass over the projects that MES did not select.

The completion rule adds those projects greedily (sorted by total supporter satisfaction,
highest first) until the budget is truly exhausted.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

from pabutools.rules.budgetallocation import BudgetAllocation
from election.md_instance import MDInstance
from pabutools.election.profile import AbstractProfile
from pabutools.election.satisfaction import SatisfactionMeasure
from pabutools.fractions import frac

from rules.md_mes.md_mes_rule import naive_md_mes


def naive_md_mes_with_completion(
    instance: MDInstance,
    profile: AbstractProfile,
    sat_class: type[SatisfactionMeasure],
    initial_budget_per_voter: list,
    projection_function: Callable,
    sat_profile=None,
) -> BudgetAllocation:
    """
    MD-MES followed by a greedy completion rule.

    Phase 1 — MES: run the standard equal-shares algorithm until no remaining
    project can be collectively afforded by its supporters.

    Phase 2 — Completion: sort the unselected projects by total summed supporter
    satisfaction (descending) and greedily add each one that fits within the
    remaining budget in every dimension.  Projects with no supporters (zero
    satisfaction) are skipped.

    Parameters
    ----------
    instance : MDInstance
    profile : AbstractProfile
    sat_class : type[SatisfactionMeasure]
    initial_budget_per_voter : list[Numeric]
        Per-voter, per-dimension budget (same as for naive_md_mes).
    projection_function : Callable
        Projection used by the MES phase (e.g. projection_function_sum).
    sat_profile : optional pre-built satisfaction profile
        Passed through to naive_md_mes.

    Returns
    -------
    BudgetAllocation
        The MES outcome extended by any greedily added projects.
    """
    # ── Phase 1: run standard MES ─────────────────────────────────────
    if sat_profile is None:
        sat_profile = profile.as_sat_profile(sat_class)

    mes_result = naive_md_mes(
        instance=instance,
        profile=profile,
        sat_class=sat_class,
        initial_budget_per_voter=initial_budget_per_voter,
        projection_function=projection_function,
        sat_profile=sat_profile,
    )

    selected_names = {p.name for p in mes_result}

    # ── Track remaining budget after MES ─────────────────────────────
    instance_dimension = next(iter(instance)).dimension
    budget_limits = list(instance.budget_limits)

    spent = [frac(0)] * instance_dimension
    for p in mes_result:
        for dim in range(instance_dimension):
            spent[dim] += p.costs[dim]

    remaining_budget = [budget_limits[dim] - spent[dim] for dim in range(instance_dimension)]

    # ── Build sat profile index for satisfaction lookup ───────────────
    # Map voter ballot -> sat object so we can compute total satisfaction
    # for each candidate project.
    total_sat_map: dict[str, float] = {}
    for p in instance:
        if p.name in selected_names:
            continue
        total = sum(
            sat_profile.multiplicity(s) * s.sat_project(p)
            for s in sat_profile
        )
        total_sat_map[p.name] = float(total)

    # ── Phase 2: greedy completion ────────────────────────────────────
    # Candidates: unselected projects with positive total satisfaction,
    # sorted by total supporter satisfaction descending so the most
    # broadly supported projects are added first.
    candidates = sorted(
        [p for p in instance if p.name not in selected_names and total_sat_map.get(p.name, 0) > 0],
        key=lambda p: (-total_sat_map[p.name], p.name),
    )

    result = BudgetAllocation(mes_result)

    for project in candidates:
        if all(
            project.costs[dim] <= remaining_budget[dim]
            for dim in range(instance_dimension)
        ):
            result.append(project)
            selected_names.add(project.name)
            for dim in range(instance_dimension):
                remaining_budget[dim] -= project.costs[dim]

    return result
