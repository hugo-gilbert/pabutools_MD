"""
Multi-dimensional Phragmén via single-dimensional projection.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from copy import deepcopy
from typing import TYPE_CHECKING

from pabutools.utils import Numeric
from pabutools.fractions import frac

from pabutools.rules.budgetallocation import BudgetAllocation
from pabutools.election import (
    Instance,
    Project,
    total_cost,
    AbstractApprovalBallot,
    AbstractApprovalProfile,
)
from pabutools.election.ballot import ApprovalBallot
from pabutools.election.profile import ApprovalProfile, AbstractProfile
from pabutools.election.satisfaction import SatisfactionMeasure
from pabutools.tiebreaking import TieBreakingRule, lexico_tie_breaking

if TYPE_CHECKING:
    from pabutools.election.md_instance import MDInstance

import logging

logger = logging.getLogger(__name__)


# =====================================================================
# Embedded single-dimensional Phragmén ("remove-and-continue" variant)
# =====================================================================


class _PhragmenVoter:
    """Summary of a voter during a run of Phragmén's sequential rule."""

    def __init__(
        self,
        ballot: AbstractApprovalBallot,
        load: Numeric,
        multiplicity: int,
        max_load: Numeric = None,
    ):
        self.ballot = ballot
        self.load = load
        self.multiplicity = multiplicity
        self.max_load = max_load

    def total_load(self):
        return self.multiplicity * self.load


def _sequential_phragmen(
    instance: Instance,
    profile: AbstractApprovalProfile,
    md_costs: dict,
    md_budget_limits: list[Numeric],
    initial_loads: list[Numeric] | None = None,
    global_max_load: Numeric | None = None,
    initial_budget_allocation: Collection[Project] | None = None,
    tie_breaking: TieBreakingRule | None = None,
    resoluteness: bool = True,
) -> BudgetAllocation | list[BudgetAllocation]:
    """
    Phragmén's sequential rule with *multi-dimensional* feasibility.

    Only applies to approval profiles.
    """
    n_dims = len(md_budget_limits)

    def fits(md_cost, project):
        """True iff adding `project` keeps every dimension within budget."""
        c = md_costs[project.name]
        return all(md_cost[d] + c[d] <= md_budget_limits[d] for d in range(n_dims))

    def add_cost(md_cost, project):
        c = md_costs[project.name]
        return [md_cost[d] + c[d] for d in range(n_dims)]

    def aux(projects, voters, alloc, md_cost, allocs):
        if len(projects) == 0:
            alloc.sort()
            if alloc not in allocs:
                allocs.append(alloc)
        else:
            min_new_maxload = None
            arg_min_new_maxload = None
            for project in projects:
                # Disqualify only if adding the project would push the TRUE
                # multi-dimensional cost over budget in some dimension.
                if not fits(md_cost, project):
                    continue
                if approval_scores[project] == 0:
                    new_maxload = float("inf")
                else:
                    new_maxload = frac(
                        sum(voters[i].total_load() for i in supporters[project])
                        + project.cost,
                        approval_scores[project],
                    )
                if min_new_maxload is None or new_maxload < min_new_maxload:
                    min_new_maxload = new_maxload
                    arg_min_new_maxload = [project]
                elif min_new_maxload == new_maxload:
                    arg_min_new_maxload.append(project)

            # Stop if every remaining project is multi-dimensionally infeasible.
            if arg_min_new_maxload is None:
                alloc.sort()
                if alloc not in allocs:
                    allocs.append(alloc)
            # Stop if selecting any project would exceed the global max load.
            elif global_max_load is not None and min_new_maxload > global_max_load:
                alloc.sort()
                if alloc not in allocs:
                    allocs.append(alloc)
            else:
                tied_projects = tie_breaking.order(
                    instance, profile, arg_min_new_maxload
                )
                if resoluteness:
                    selected_project = tied_projects[0]
                    for voter in voters:
                        if selected_project in voter.ballot:
                            voter.load = min_new_maxload
                    alloc.append(selected_project)
                    projects.remove(selected_project)
                    aux(
                        projects,
                        voters,
                        alloc,
                        add_cost(md_cost, selected_project),
                        allocs,
                    )
                else:
                    for selected_project in tied_projects:
                        new_voters = deepcopy(voters)
                        for voter in new_voters:
                            if selected_project in voter.ballot:
                                voter.load = min_new_maxload
                        new_alloc = deepcopy(alloc) + [selected_project]
                        new_md_cost = add_cost(md_cost, selected_project)
                        new_projs = deepcopy(projects)
                        new_projs.remove(selected_project)
                        aux(new_projs, new_voters, new_alloc, new_md_cost, allocs)

    if not isinstance(profile, AbstractApprovalProfile):
        raise ValueError("The Sequential Phragmen Rule only applies to approval profiles.")

    if tie_breaking is None:
        tie_breaking = lexico_tie_breaking
    if initial_budget_allocation is None:
        initial_budget_allocation = BudgetAllocation()
    else:
        initial_budget_allocation = BudgetAllocation(initial_budget_allocation)

    current_md_cost = [0] * n_dims
    for p in initial_budget_allocation:
        current_md_cost = add_cost(current_md_cost, p)

    # A project is eligible from the start only if it is feasible on its own in
    # every dimension of the true multi-dimensional budget.
    initial_projects = set(
        p
        for p in instance
        if p not in initial_budget_allocation
        and all(md_costs[p.name][d] <= md_budget_limits[d] for d in range(n_dims))
    )

    if initial_loads is None:
        voters_details = [_PhragmenVoter(b, 0, profile.multiplicity(b)) for b in profile]
    else:
        voters_details = [
            _PhragmenVoter(b, initial_loads[i], profile.multiplicity(b))
            for i, b in enumerate(profile)
        ]
    supporters = {
        proj: [i for i, v in enumerate(voters_details) if proj in v.ballot]
        for proj in initial_projects
    }

    approval_scores = {project: profile.approval_score(project) for project in instance}

    all_budget_allocations: list[BudgetAllocation] = []
    aux(
        initial_projects,
        voters_details,
        initial_budget_allocation,
        current_md_cost,
        all_budget_allocations,
    )

    if resoluteness:
        return all_budget_allocations[0]
    return all_budget_allocations


# =====================================================================
# Public entry point: same signature/output shape as naive_md_mes
# =====================================================================


def projected_md_phragmen(
    instance: "MDInstance",
    profile: AbstractProfile,
    sat_class: type[SatisfactionMeasure],
    initial_budget_per_voter: list[Numeric],
    projection_function: Callable,
    sat_profile=None,
    log_path: str | None = None,
) -> BudgetAllocation:
    """
    Multi-dimensional Phragmén by single-dimensional projection.

    Parameters
    ----------
        instance: MDInstance
            The multi-dimensional instance.
        profile: AbstractProfile
            The profile. Each ballot is interpreted as the set of projects it
            contains (approval semantics). Assumes one ballot per voter, as the
            pipeline's parser produces.
        sat_class: type[SatisfactionMeasure]
            Ignored. Present only for signature compatibility with
            ``naive_md_mes``.
        initial_budget_per_voter: list[Numeric]
            Ignored. Present only for signature compatibility.
        projection_function: Callable
            An L^p norm ``g`` mapping a cost vector to a scalar. Applied to each
            project's cost vector to obtain its single-dimensional cost.
        sat_profile: optional
            Ignored. Present only for signature compatibility.
        log_path: str, optional
            Ignored. Present only for signature compatibility.

    Returns
    -------
        BudgetAllocation
            The selected projects, as the original multi-dimensional projects
            (same format returned by ``naive_md_mes``).
    """
    # 1. Project every multi-dimensional project to a single scalar cost g(c(p)).
    orig_by_name: dict[str, object] = {}
    sd_project_by_name: dict[str, Project] = {}
    for p in instance:
        orig_by_name[p.name] = p
        scalar_cost = projection_function([float(c) for c in p.costs])
        sd_project_by_name[p.name] = Project(name=p.name, cost=scalar_cost)

    # 2. Single-dimensional budget = the per-dimension budget of the MD instance.
    #    The projected budget still parameterises the
    #    single-dimensional instance; feasibility is enforced against the true
    #    multi-dimensional budget below.
    budget = instance.budget_limits[0]
    sd_instance = Instance(sd_project_by_name.values(), budget_limit=budget)

    # 3. Rebuild an approval profile over the single-dimensional projects.
    sd_ballots = [
        ApprovalBallot(
            sd_project_by_name[p.name] for p in ballot if p.name in sd_project_by_name
        )
        for ballot in profile
    ]
    sd_profile = ApprovalProfile(sd_ballots)

    # 4. Run Phragmén with true multi-dimensional feasibility: loads/order use
    #    the projected cost, but a project is only disqualified when adding it
    #    would exceed the real per-dimension budget.
    md_costs = {name: list(p.costs) for name, p in orig_by_name.items()}
    md_budget_limits = list(instance.budget_limits)
    sd_outcome = _sequential_phragmen(sd_instance, sd_profile, md_costs, md_budget_limits)

    # 5. Map back to the original MD projects; same output format as naive_md_mes.
    res = BudgetAllocation()
    for sp in sd_outcome:
        res.append(orig_by_name[sp.name])
    return res
