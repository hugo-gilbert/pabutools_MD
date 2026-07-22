"""
Module defining the basic classes used to represent a multidimentional participatory budgeting election.
The :py:class:`~pabutools.election.md_instance.MDProject` and the
:py:class:`~pabutools.election.md_instance.MDInstance` classes are defined here.
"""

from __future__ import annotations

from collections.abc import Collection, Generator

from pulp import LpProblem, LpMaximize, LpVariable, LpBinary, lpSum, LpStatusOptimal, value, HiGHS

from pabutools.utils import Numeric

from pabutools.fractions import frac
from pabutools.utils import powerset
from pabutools.election.instance import Project, Instance

from math import ceil

import pulp

import random


class MDProject:
    """
    Represents a project, that is, the entity that is voted upon in a participatory budgeting election.

    Parameters
    ----------
        name : str, optional
            The name of the project. This is used as the identifier of a project. It should be unique with a collection
            of projects, though this is not enforced.
            Defaults to `""`.
        dimension : int, optional
            The dimension of the PB election
            Defaults to 1
        costs : list[Numeric], optional
            The costs of the project of size dimension.
            Defaults to `[0]`.
        categories: set[str], optional
            The categories that the project is a member of. These categories can  "Urban greenery" or "Public
            transport" for instance.
            Defaults to `{}`.
        targets: set[str], optional
            The target groups that the project is targeting. These can be "Citizens above 60 years old" or
            "Residents of district A" for instance.
            Defaults to `{}`.

    Attributes
    ----------
        name : str
            The name of the project. This is used as the identifier of a project. It should be unique with a collection
            of projects, though this is not enforced.
        dimension : int
            The dimension of the PB election.
        costs : list[Numeric]
            The costs of the project, of size dimension.
        categories: set[str]
            The categories that the project is a member of. These categories can  "Urban greenery" or "Public
            transport" for instance.
        targets: set[str]
            The target groups that the project is targeting. These can be "Citizens above 60 years old" or
            "Residents of district A" for instance.
    """

    def __str__(self) -> str:
        return str(self.name)

    def __repr__(self) -> str:
        return self.__str__() 

    def __init__(
        self, name: str = "", dimension: int = 1, costs: list[Numeric] = [0], categories=None, targets=None
    ) -> None:
        if targets is None:
            targets = {}
        if categories is None:
            categories = {}
        self.name = name
        self.dimension = dimension
        self.costs = list(costs)
        for i in range(self.dimension):
            if not int(costs[i]) == costs[i]:
                self.costs[i] = frac(costs[i])  # float costs do not work, enforce fractions
            else:
                self.costs[i] = int(costs[i])
        try:
            if not (self.dimension == len(costs)): 
                raise Exception()
        except Exception:
            print("costs does not have size dimension")
        self.categories = categories
        self.targets = targets

    def sum_projection(self) -> Project:
        return Project(name=self.name+"sum", cost = sum(self.costs))

    def max_projection(self) -> Project:
        return Project(name=self.name+"max", cost = max(self.costs))

    def normalize(self, budgets : list[Numeric]) -> None:
        for i in range(self.dimension):
            self.costs[i]=frac(self.costs[i],budgets[i])


    def __eq__(self, other) -> bool:
        if isinstance(other, MDProject):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other
        return False

    def __le__(self, other) -> bool:
        if isinstance(other, MDProject):
            return self.name.__le__(other.name)
        if isinstance(other, str):
            return self.name.__le__(other)
        return False

    def __lt__(self, other) -> bool:
        if isinstance(other, MDProject):
            return self.name.__lt__(other.name)
        if isinstance(other, str):
            return self.name.__lt__(other)
        return False

    def __hash__(self) -> int:
        return hash(self.name)



def dimension(projects: Collection[MDProject]) -> int:
    d = min(p.dimension for p in projects)
    try:
        for p in projects:
            if not (p.dimension == d): 
                raise Exception()
    except Exception:
        print("projects do not all have the same dimension")
    return d

def total_costs(projects: Collection[MDProject]) -> list[Numeric]:
    """
    Returns the total costs of a collection of projects, summing the cost of the projects for each type of ressource.

    Parameters
    ----------
        projects : iterable[:py:class:`~pabutools.election.md_instance.MDProject`]
            An iterable of projects.

    Returns
    -------
        list[Numeric]
            The total costs for each type of ressource
    """
    d = dimension(projects)
    return [sum(p.costs[i] for p in projects) for i in range(d)]


def max_budget_allocation_cardinality(
    projects: Collection[MDProject], budget_limits: list[Numeric]
) -> int:
    """
    Returns the maximum number of projects that can be chosen with respect to the budget limits.

    Parameters
    ----------
        projects : iterable[:py:class:`~pabutools.election.md_instance.MDProject`]
            An iterable of projects.
        budget_limits : list[Numeric]
            the budget limits

    Returns
    -------
        int
            The maximum number of projects that can be chosen with respect to the budget limits.

    """
    mip_model = LpProblem("MaxCardinalityAllocation", LpMaximize)

    p_vars = {p: LpVariable(f"x_{p}", cat=LpBinary) for p in projects}

    if not p_vars:
        return 0


    d = dimension(projects)

    # Objective: maximize total cardinality of selected projects        
    mip_model += lpSum(p_vars[p] for p in projects)

    # Budget constraint
    for i in range(d):
        mip_model += lpSum(p_vars[p] * p.costs[i] for p in projects) <= budget_limits[i]

    # Solve the model
    solver = pulp.PULP_CBC_CMD()
    mip_model.solve(solver)#HiGHS(msg=False))

    if mip_model.status == LpStatusOptimal:
        max_cost = value(mip_model.objective)
        return frac(float(max_cost))


def max_budget_allocation_sum_costs(
    projects: Collection[MDProject], budget_limits: list[Numeric]
) -> Numeric:
    """
    Returns the maximum total sum of costs over all subsets of projects with respect to the budget limits.

    Parameters
    ----------
        projects : iterable[:py:class:`~pabutools.election.md_instance.MDProject`]
            An iterable of projects.
        budget_limits : list[Numeric]
            the budget limits

    Returns
    -------
        int
            The maximum total cost over all subsets of projects with respect to the budget limits.

    """
    mip_model = LpProblem("MaxCostAllocation", LpMaximize)

    p_vars = {p: LpVariable(f"x_{p}", cat=LpBinary) for p in projects}

    if not p_vars:
        return 0

    d = dimension(projects)
    
    # Objective: maximize total cost of selected projects  
    mip_model += lpSum(p_vars[p] * p.costs[i] for p in projects for i in range(d))

    # Budget constraint
    for i in range(d):
        mip_model += lpSum(p_vars[p] * p.costs[i] for p in projects) <= budget_limits[i]

    # Solve the model
    solver = pulp.PULP_CBC_CMD()
    mip_model.solve(solver)
    #mip_model.solve(HiGHS(msg=False))

    if mip_model.status == LpStatusOptimal:
        max_cost = value(mip_model.objective)
        return frac(float(max_cost))


class MDInstance(set[MDProject]):
    """
    Participatory budgeting instances.
    An instance contains the projects that are voted on, together with other information about the election such as the
    budget limits.
    Importantly, the ballots submitted by the voters is not part of the instance.
    See the module :py:mod:`~pabutools.election.profile` for how to handle the voters.
    Note that `MDInstance` is a subclass of the Python class `set`, and can be used as a set is.

    Parameters
    ----------
        init: Iterable[:py:class:`~pabutools.election.md_instance.MDProject`], optional
            An iterable of :py:class:`~pabutools.election.md_Instance.MDProject` that constitutes the initial set of projects
            for the instance. In case an :py:class:`~pabutools.election.mDInstance.Instance` object is passed, the
            additional attributes are also copied (except if the corresponding keyword arguments have been given).
        budget_limits : list[Numeric], optional
            The budget limits of the instance, that is, the maximum amount of money a set of projects can use to be
            feasible for different types of resource.
        categories: set[str], optional
            The set of categories that the projects can be assigned to. These can be "Urban greenery" or "Public
            transport" for instance.
            Defaults to `{}`.
        targets: set[str], optional
            The set of target groups that the project can be targeting. These can be "Citizens above 60 years old" or
            "Residents of district A" for instance.
            Defaults to `{}`.
        file_path : str, optional
            If the instance has been parsed from a file, the path to the file.
            Defaults to `""`.
        file_name : str, optional
            If the instance has been parsed from a file, the name of the file.
            Defaults to `""`.
        parsing_errors : bool, optional
            Boolean indicating if errors were encountered when parsing the file.
            Defaults to `None`.
        meta : dict, optional
            All kinds of relevant information for the instance, stored in a dictionary. Keys and values are
            typically strings.
            Defaults to `dict()`.
        project_meta : dict[:py:class:`~pabutools.election.md_instance.MDProject`, dict], optional
            All kinds of relevant information about the projects, stored in a dictionary. Keys are
            :py:class:`~pabutools.election.md_instance.MDProject` and values are dictionaries.
            Defaults to `dict()`.


    Attributes
    ----------
        budget_limits : list[Numeric]
            The budget limits of the instance, that is, the maximum amount of money a set of projects can use to be
            feasible for different types of resources.
        categories: set[str]
            The set of categories that the projects can be assigned to. These can be "Urban greenery" or "Public
            transport" for instance.
        targets: set[str]
            The set of target groups that the project can be targeting. These can be "Citizens above 60 years old" or
            "Residents of district A" for instance.
        file_path : str
            If the instance has been parsed from a file, the path to the file.
        file_name : str
            If the instance has been parsed from a file, the name of the file.
        parsing_errors : bool
            Boolean indicating if errors were encountered when parsing the file.
        meta : dict
            All kinds of relevant information for the instance, stored in a dictionary. Keys and values are
            typically strings.
        project_meta : dict[:py:class:`~pabutools.election.md_instance.MDProject`: dict]
            All kinds of relevant information about the projects, stored in a dictionary. Keys are
            :py:class:`~pabutools.election.md_instance.MDProject` and values are dictionaries.
    """

    def __init__(
        self,
        init: Collection[MDProject] = (),
        budget_limits: list[Numeric] | None = None,
        categories: set[str] | None = None,
        targets: set[str] | None = None,
        file_path: str | None = None,
        file_name: str | None = None,
        parsing_errors: bool | None = None,
        meta: dict | None = None,
        project_meta: dict | None = None,
    ) -> None:
        set.__init__(self, init)

        self.budget_limits: list[Numeric] = (
            [0]  # Only for type checking, so that init.budget_limit does not fail
        )
        if budget_limits is None:
            if isinstance(init, MDInstance):
                budget_limits = list(init.budget_limits)
            else:
                budget_limits = [0]
        self.budget_limits = list(budget_limits)

        self.categories = (
            None  # Only for type checking, so that init.categories does not fail
        )
        if categories is None:
            if isinstance(init, MDInstance):
                categories = init.categories
            else:
                categories = set()
        self.categories = categories

        self.targets = (
            None  # Only for type checking, so that init.targets does not fail
        )
        if targets is None:
            if isinstance(init, MDInstance):
                targets = init.targets
            else:
                targets = set()
        self.targets = targets

        self.file_path = (
            None  # Only for type checking, so that init.file_path does not fail
        )
        if file_path is None:
            if isinstance(init, MDInstance):
                file_path = init.file_path
            else:
                file_path = ""
        self.file_path = file_path

        self.file_name = (
            None  # Only for type checking, so that init.file_name does not fail
        )
        if file_name is None:
            if isinstance(init, MDInstance):
                file_name = init.file_name
            else:
                file_name = ""
        self.file_name = file_name

        self.parsing_errors = (
            None  # Only for type checking, so that init.parsing_errors does not fail
        )
        if parsing_errors is None:
            if isinstance(init, MDInstance):
                parsing_errors = init.parsing_errors
            else:
                parsing_errors = False
        self.parsing_errors = parsing_errors

        self.meta = None  # Only for type checking, so that init.meta does not fail
        if meta is None:
            if isinstance(init, MDInstance):
                meta = init.meta
            else:
                meta = dict()
        self.meta = meta

        self.project_meta = (
            None  # Only for type checking, so that init.projet_meta does not fail
        )
        if project_meta is None:
            if isinstance(init, MDInstance):
                project_meta = init.project_meta
            else:
                project_meta = dict()
        self.project_meta = project_meta

    def get_project(self, project_name: str) -> MDProject:
        """
        Iterates over the instance to find a project with the given name. If found, the project is returned, otherwise
        a `KeyError` exception is raised.

        Returns
        -------
            :py:class:`~pabutools.election.md_instance.MDProject`
                The project.
        """
        for p in self:
            if p.name == project_name:
                return p
        raise KeyError(
            "No project with name {} found in the instance.".format(project_name)
        )

    def budget_allocations(self) -> Generator[Collection[MDProject]]:
        """
        Returns a generator for all the feasible budget allocations of the instance.

        Returns
        -------
            Generator[Iterable[:py:class:`~pabutools.election.md_instance.MDProject`]
                The generator.

        """
        for b in powerset(self):
            if self.is_feasible(b):
                yield b

    def is_trivial(self) -> bool:
        """
        Tests if the instance is trivial, meaning that either all projects can be selected without
        exceeding the budget limit, or that no project can be selected.

        Returns
        -------
            bool
                `True` if the instance is trivial, `False` otherwise.
        """
        d = dimension(self)
        try:
            if not(len(self.budget_limits) == d):
                raise Exception()
        except Exception:
            print("projects' dimension does not match budget limits' size")
        return all([total_costs(self)[i] <= self.budget_limits[i] for i in range(d)]) or any(
            [self.budget_limits[i] <= min(p.costs[i] for p in self) for i in range(d)])

    def is_feasible(self, projects: Collection[MDProject]) -> bool:
        """
        Tests if a given collection of projects is feasible for the instance, meaning that the total cost of the
        projects does not exceed the budget limits of the instance.

        Parameters
        ----------
            projects : Iterable[:py:class:`~pabutools.election.md_instance.MDProject`]
                The collection of projects.
        Returns
        -------
            bool
                `True` if the collection of project costs less than the budget limits for all limits, `False` otherwise.
        """
        d = dimension(projects)
        try:
            if not(len(self.budget_limits) == d):
                raise Exception()
        except Exception:
            print("projects' dimension does not match budget limits' size")
        return all([total_costs(projects)[i] <= self.budget_limits[i] for i in range(d)])

    def is_exhaustive(
        self,
        projects: Collection[MDProject],
        available_projects: Collection[MDProject] | None = None,
    ) -> bool:
        """
        Tests if a given collection of projects is exhaustive. A collection of projects is said to be exhaustive if no
        additional project could be added without violating the budget limit.
        Note that a collection of projects can be exhaustive, but not feasibility.

        Parameters
        ----------
            projects : Iterable[:py:class:`~pabutools.election.md_instance.MDProject`]
                The collection of projects.
            available_projects : Iterable[:py:class:`~pabutools.election.md_instance.MDProject`], optional
                Only these projects are considered when testing for exhaustiveness. Defaults to None, i.e., considering
                all projects.
        Returns
        -------
            bool
                `True` if the collection of project is exhaustive, `False` otherwise.
        """
        if available_projects is None:
            available_projects = self
        
        d = dimension(projects)
        d2 = dimension(available_projects)
        try:
            if not(len(self.budget_limits) == d) or not(d2 == d):
                raise Exception()
        except Exception:
            print("projects' dimension does not match budget limits' size or available_projects' dimension")
        costs = total_costs(projects)
        for p in available_projects:
            if p not in projects and all([p.costs[i] + costs[i] <= self.budget_limits[i] for i in range(d)]):
                return False
        return True

    def normalize(self) -> None:
        for project in self:
            project.normalize(self.budget_limits)
        for i in range(len(self.budget_limits)):
            self.budget_limits[i]=1

    def sum_projection(self) -> Instance:
        self.normalize()
        projects = []
        for project in self:
            projects.append(project.sum_projection())
        return Instance(init = projects, budget_limit=1)

    def max_projection(self) -> Instance:
        self.normalize()
        projects = []
        for project in self:
            projects.append(project.max_projection())
        return Instance(init = projects, budget_limit=1)

    def __str__(self) -> str:
        res = "MDInstance "
        if self.file_name:
            res += "({}) ".format(self.file_name)
        res += "with budget limits {} and {} projects:\n".format(
            self.budget_limits, len(self)
        )
        for p in self:
            res += "\tc({}) = {}\n".format(p, p.costs)
        return res[:-1]

    def __repr__(self) -> str:
        return self.__str__()

    # This allows set method returning copies of a set to work with PBInstances
    @classmethod
    def _wrap_methods(cls, names):
        def wrap_method_closure(name):
            def inner(self, *args):
                result = getattr(super(cls, self), name)(*args)
                if isinstance(result, set) and not isinstance(result, cls):
                    result = cls(
                        result,
                        budget_limits=self.budget_limits,
                        categories=self.categories,
                        targets=self.targets,
                        file_name=self.file_name,
                        file_path=self.file_path,
                        parsing_errors=self.parsing_errors,
                        meta=self.meta,
                        project_meta=self.project_meta,
                    )
                return result

            inner.fn_name = name
            setattr(cls, name, inner)

        for n in names:
            wrap_method_closure(n)


MDInstance._wrap_methods(
    [
        "__ror__",
        "difference_update",
        "__isub__",
        "symmetric_difference",
        "__rsub__",
        "__and__",
        "__rand__",
        "intersection",
        "difference",
        "__iand__",
        "union",
        "__ixor__",
        "symmetric_difference_update",
        "__or__",
        "copy",
        "__rxor__",
        "intersection_update",
        "__xor__",
        "__ior__",
        "__sub__",
    ]
)



def get_random_instance(num_projects: int, min_cost: int, max_cost: int, dimension: int) -> MDInstance:
    """
    Generates a random instance. Costs and budget limit are integers. The costs are selected uniformly at random between
    `min_cost` and `max_cost`. The budget limit is sample form a uniform between the minimum cost of a project
    and the total cost of all the projects.
    The parameters are rounded up to the closest int.

    Parameters
    ----------
        num_projects : int
            The number of projects in the instance.
        min_cost : int
            The minimum cost of a project for any type of ressource.
        max_cost : int
            The maximum cost of a project for any type of ressource.
        dimension : int
            The dimension of the instance
    Returns
    -------
        pabutools.election.md_instance.MDInstance
            The randomly-generated instance.
    """
    inst = MDInstance()
    inst.update(
        MDProject(
            name=str(p),
            costs=[random.randint(round(min_cost), round(max_cost)) for i in range(dimension)],
            dimension = dimension,
        )
        for p in range(round(num_projects))
    )
    inst.budget_limits = [random.randint(
        ceil(max(p.costs[i] for p in inst)), ceil(sum(p.costs[i] for p in inst))
    ) for i in range(dimension)]
    return inst
