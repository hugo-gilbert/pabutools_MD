"""
Greedy total-satisfaction selection rule.
"""

from __future__ import annotations

from collections.abc import Callable

from pabutools.rules.budgetallocation import BudgetAllocation
from election.md_instance import MDInstance
from pabutools.election.profile import AbstractProfile
from pabutools.election.satisfaction import SatisfactionMeasure


def naive_md_greedy(
    instance: MDInstance,
    profile: AbstractProfile,
    sat_class: type[SatisfactionMeasure],
    initial_budget_per_voter: list,
    projection_function: Callable,
    sat_profile=None,
    log_path: str | None = None,
) -> BudgetAllocation:
    """
    Greedy selection by total summed supporter satisfaction.
    """
    if sat_profile is None:
        sat_profile = profile.as_sat_profile(sat_class)

    instance_dimension = next(iter(instance)).dimension
    remaining_budget = list(instance.budget_limits)

    # ── Total summed satisfaction across all approving agents, per project ──
    # For each ballot, its multiplicity times the satisfaction it derives from
    # the project; summed over all ballots (non-approvers contribute 0).
    total_sat_map: dict[str, float] = {}
    for p in instance:
        total = sum(
            sat_profile.multiplicity(s) * s.sat_project(p)
            for s in sat_profile
        )
        total_sat_map[p.name] = float(total)

    # ── Ordering: highest total satisfaction first, ties broken by name ──────
    candidates = sorted(
        [p for p in instance if total_sat_map.get(p.name, 0) > 0],
        key=lambda p: (-total_sat_map[p.name], p.name),
    )

    # ── Greedy first-fit pass against the per-dimension budget ───────────────
    result = BudgetAllocation()
    for project in candidates:
        if all(
            project.costs[dim] <= remaining_budget[dim]
            for dim in range(instance_dimension)
        ):
            result.append(project)
            for dim in range(instance_dimension):
                remaining_budget[dim] -= project.costs[dim]

    return result
