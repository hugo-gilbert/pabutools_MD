"""
Multi-dimensional method of equal shares.
"""

from __future__ import annotations

from copy import copy, deepcopy
from collections.abc import Iterable

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
    """
    Class used to summarise a voter during a run of the multi-dimensional method of equal shares.

    Parameters
    ----------
        index: Numeric
            The index of the voter in the voter list
        ballot: :py:class:`~pabutools.election.ballot.ballot.AbstractBallot`
            The ballot of the voter.
        sat: SatisfactionMeasure
            The satisfaction measure corresponding to the ballot.
        budget: list[Numeric]
            The budgets of the voter across each dimension.
        multiplicity: int
            The multiplicity of the ballot.

    Attributes
    ----------
        index: int
            The index of the voter in the list of voters MES maintains
        ballot: :py:class:`~pabutools.election.ballot.ballot.AbstractBallot`
            The ballot of the voter.
        sat: SatisfactionMeasure
            The satisfaction measure corresponding to the ballot.
        budget: list[Numeric]
            The budgets of the voter across each dimension.
        multiplicity: int
            The multiplicity of the ballot.
        budget_over_sat_map: dict[list[Numeric], Numeric]
            Maps values of the budget to values of the budget divided by the total satisfaction.
    """

    def __init__(
        self,
        index: Numeric,
        ballot: AbstractBallot,
        sat: SatisfactionMeasure,
        budget: list[Numeric],
        multiplicity: int,
    ):
        self.index: int = index
        self.ballot: AbstractBallot = ballot
        self.sat: SatisfactionMeasure = sat
        self.budget: list[Numeric] = budget
        self.multiplicity: int = multiplicity
        self.budget_over_sat_map: dict[tuple[Project, list[Numeric]], Numeric] = dict()

    def total_sat_project(self, proj: Project) -> Numeric:
        """
        Returns the total satisfaction of a given project. It is equal to the satisfaction for the project,
        multiplied by the multiplicity.

        Parameters
        ----------
            proj: :py:class:`~pabutools.election.instance.Project`
                The project.

        Returns
        -------
            Numeric
                The total satisfaction.
        """
        return self.multiplicity * self.sat.sat_project(proj)

    def total_budget(self) -> list[Numeric]:
        """
        Returns the total budget of the voters. It is equal to the budget multiplied by the multiplicity.

        Returns
        -------
            list[Numeric]
                list of the total budgets.
        """
        return [self.multiplicity*budget_x for budget_x in self.budget]

    def budget_over_sat_project(self, proj):
        """
        Returns the list of budgets divided by the satisfaction for a given project.

        Parameters
        ----------
            proj: :py:class:`~pabutools.election.instance.Project`
                The collection of projects.

        Returns
        -------
            list[Numeric]
                The list of budgets, each divided by the satisfaction
        """
        res = self.budget_over_sat_map.get((proj, self.budget), None)
        if res is None:
            res = [frac(budget_x, self.sat.sat_project(proj)) for budget_x in self.budget]
            self.budget_over_sat_map[(proj, self.budget)] = res
        return res

    def __str__(self):
        return f"MESVoter[{self.budget}]"

    def __repr__(self):
        return f"MESVoter[{self.budget}]"

class MDMESProject(MDProject):
    """
    Class used to summarise the projects in a run of multi-dimensional MES. Mostly use to store details that can be retrieved
    efficiently.
    """

    def __init__(self, project):
        MDProject.__init__(self, project.name, project.dimension, project.costs)
        self.project = project
        self.total_sat = None
        self.sat_supporter_map = dict()
        self.unique_sat_supporter = None
        self.supporter_indices = []
        self.initial_affordability = None
        self.affordability = None

    def supporters_sat(self, supporter: MDMESVoter):
        if self.unique_sat_supporter:
            return self.unique_sat_supporter
        return supporter.sat.sat_project(self)

    def __str__(self):
        return f"MESProject[{self.name}, {float(self.affordability)}]"

    def __repr__(self):
        return f"MESProject[{self.name}, {float(self.affordability)}]"

def affordability_poor_rich_vector(voters: list[MDMESVoter], project: MDMESProject) -> list[Numeric]:
    """Compute the affordability factor of a project using the "poor/rich" algorithm.
       
    Parameters
    ----------
        voters: list[MDMESVoter]
            The list of the voters, formatted for MDMES.
        project: MDMESProject
            The project under consideration.

    Returns
    -------
        Numeric
            The vector of affordability factors of the project.

    """
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
                i
                for i in rich
                if voters[i].total_budget()[dim]
                < affordability * voters[i].sat.sat_project(project)
            }
            if len(new_poor) == 0:
                break
            rich -= new_poor
            poor.update(new_poor)
        affordability_vector.append(affordability)
    return affordability_vector

def projection_function_max(affordability_vector: list[Numeric]) -> Numeric:
    """
    Returns the maximum value from an affordability vector

    Parameters
    ----------
        affordability_vector: list[Numeric]
            The affordability vector

    Returns
    ----------
        Numeric
            The maximum value of the vector
    """
    return max(affordability_vector)

def projection_function_sum(affordability_vector: list[Numeric]) -> Numeric:
    """
    Returns the summed value from an affordability vector

    Parameters
    ----------
        affordability_vector: list[Numeric]
            The affordability vector

    Returns
    ----------
        Numeric
            The summed value of the vector
    """
    return sum(affordability_vector)

def naive_md_mes(
    instance: MDInstance,
    profile: AbstractProfile,
    sat_class: type[SatisfactionMeasure],
    initial_budget_per_voter: list[Numeric],
    projection_function: Callable[list[Numeric], Numeric]
) -> BudgetAllocation:
    """
    Naive implementation of the method of equal shares. Probably slow, but useful to test the
    correctness of other implementations.

    Parameters
    ----------
        instance: Instance
            The instance.
        profile: AbstractProfile
            The profile.
        sat_class: type[SatisfactionMeasure]
            The satisfaction measure used as a proxy of the satisfaction of the voters.
        initial_budget_per_voter: list[Numeric]
            The initial budget allocated to the voters in the run of MES.
        projection_function: Callable[list[Numeric], Numeric]
            A function that turns a vector of affordability parameters into a single value

    Returns
    -------
        BudgetAllocation
            All the projects selected by the method of equal shares.

    """
    sat_profile = profile.as_sat_profile(sat_class)
    voters = []
    
    #there may be a better way of doing this, should we store the dimension inside the MD-instance object directly?
    instance_dimension = next(iter(instance)).dimension

    for index, sat in enumerate(sat_profile):
        voters.append(
            MDMESVoter(
                index,
                sat.ballot,
                sat,
                deepcopy(initial_budget_per_voter),
                sat_profile.multiplicity(sat),
            )
        )
        index += 1

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
        if total_sat > 0:
            if sum(p.costs) > 0:
                mes_p.total_sat = total_sat
                projects.add(mes_p)

    res = BudgetAllocation()
    affordabilities_projections = dict()
    affordabilities_vectors = dict()

    remaining_projects = deepcopy(projects)
    while True:
        to_remove = set()
        for project in remaining_projects:
            if (
                any(sum(voters[i].total_budget()[dim] for i in project.supporter_indices) < project.costs[dim]
                for dim in range(instance_dimension))
            ):
                to_remove.add(project)
            afford_vector = affordability_poor_rich_vector(voters, project)
            if afford_vector is not None:
                affordabilities_vectors[project] = afford_vector
                affordabilities_projections[project] = projection_function(afford_vector)
        for project in to_remove:
            remaining_projects.remove(project)
            if project in affordabilities_vectors:
                del affordabilities_vectors[project]
                del affordabilities_projections[project]
        if len(remaining_projects) == 0:
            return res
        min_afford = min(affordabilities_projections.values())
        selected = [p for p in remaining_projects if affordabilities_projections[p] == min_afford][
            0
        ]
        res.append(selected.project)
        remaining_projects.remove(selected)
        for i in selected.supporter_indices:
            for dim in range(instance_dimension):
                voters[i].budget[dim] -= min(
                    voters[i].budget[dim], affordabilities_vectors[selected][dim] * voters[i].sat.sat_project(selected)
                )
        del affordabilities_vectors[selected]
        del affordabilities_projections[selected]