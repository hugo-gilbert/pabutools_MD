from __future__ import annotations

from collections.abc import Callable, Collection

from pulp import LpProblem, LpMaximize, LpBinary, LpVariable, lpSum, value, HiGHS

from pabutools.utils import Numeric
from pabutools.election.satisfaction.additivesatisfaction import AdditiveSatisfaction

import numpy as np

from pabutools.election.satisfaction.satisfactionmeasure import SatisfactionMeasure
from pabutools.election.ballot import (
    AbstractBallot,
    AbstractApprovalBallot,
    AbstractCardinalBallot,
)
from pabutools.election.instance import (
    Instance,
    Project,
    total_cost,
    max_budget_allocation_cardinality,
    max_budget_allocation_cost,
)
from pabutools.election.md_instance import (MDInstance, MDProject)

from pabutools.fractions import frac

from typing import TYPE_CHECKING

def max_cost_sat_func(
    instance: MDInstance,
    profile: AbstractProfile,
    ballot: AbstractBallot,
    project: MDProject,
    precomputed_values: dict,
) -> int:
    """
    Computes the max cost satisfaction for ballots. It is equal to the maximum cost of the project along its dimensions if it appears in the
    ballot and 0 otherwise.

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.md_instance.MDInstance`
            The instance.
        profile : :py:class:`~pabutools.election.profile.profile.AbstractProfile`
            The profile.
        ballot : :py:class:`~pabutools.election.ballot.ballot.AbstractBallot`
            The ballot.
        project : :py:class:`~pabutools.election.md_instance.MDProject`
            The selected project.
        precomputed_values : dict[str, str]
            A dictionary of precomputed values.

    Returns
    -------
        int
            The max cost satisfaction.
    """
    return int(project in ballot) * max(project.costs)

class Max_Cost_Sat(AdditiveSatisfaction):
    """
    The max cost satisfaction for ballots. It is equal to the maximum total cost across dimensions of the selected projects appearing in the ballot.
    It applies to all ballot types that support the `in` operator.

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.MDInstance`
            The instance.
        profile : :py:class:`~pabutools.election.profile.profile.AbstractProfile`
            The profile.
        ballot : :py:class:`~pabutools.election.ballot.ballot.AbstractBallot`
            The ballot.
    """

    def __init__(
        self, instance: MDInstance, profile: AbstractProfile, ballot: AbstractBallot
    ):
        AdditiveSatisfaction.__init__(self, instance, profile, ballot, max_cost_sat_func)

def sum_cost_sat_func(
    instance: MDInstance,
    profile: AbstractProfile,
    ballot: AbstractBallot,
    project: MDProject,
    precomputed_values: dict,
) -> int:
    """
    Computes the sum cost satisfaction for ballots. It is equal to the summed cost of the project across its dimensions if it appears in the
    ballot and 0 otherwise.

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.md_instance.MDInstance`
            The instance.
        profile : :py:class:`~pabutools.election.profile.profile.AbstractProfile`
            The profile.
        ballot : :py:class:`~pabutools.election.ballot.ballot.AbstractBallot`
            The ballot.
        project : :py:class:`~pabutools.election.md_instance.MDProject`
            The selected project.
        precomputed_values : dict[str, str]
            A dictionary of precomputed values.

    Returns
    -------
        int
            The sum cost satisfaction.
    """
    return int(project in ballot) * sum(project.costs)

class Sum_Cost_Sat(AdditiveSatisfaction):
    """
    The sum cost satisfaction for ballots. It is equal to the total summed cost of the selected projects appearing in the ballot.
    It applies to all ballot types that support the `in` operator.

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.md_instance.MDInstance`
            The instance.
        profile : :py:class:`~pabutools.election.profile.profile.AbstractProfile`
            The profile.
        ballot : :py:class:`~pabutools.election.ballot.ballot.AbstractBallot`
            The ballot.
    """

    def __init__(
        self, instance: MDInstance, profile: AbstractProfile, ballot: AbstractBallot
    ):
        AdditiveSatisfaction.__init__(self, instance, profile, ballot, sum_cost_sat_func)