"""
Multidimensional Tools to work with PaBuLib.
Random simplex split variant: each project's cost is randomly distributed
across ALL of its categories (not just the first one).
"""

import numpy as np
from copy import deepcopy

from pabutools.fractions import str_as_frac
from pabutools.election.md_instance import MDInstance, MDProject
from pabutools.election.ballot import (
    ApprovalBallot,
    CardinalBallot,
    OrdinalBallot,
    CumulativeBallot,
)
from pabutools.election.profile import (
    Profile,
    ApprovalProfile,
    CardinalProfile,
    CumulativeProfile,
    OrdinalProfile,
)

import csv
import os


def _sample_simplex(n, rng=None):
    """
    Sample a point uniformly at random from the (n-1)-dimensional simplex.
    Uses the sorted-uniforms method: draw n-1 uniforms, sort them, and take
    consecutive differences.

    Returns a numpy array of length n that sums to 1.
    """
    if rng is None:
        rng = np.random.default_rng()
    if n == 1:
        return np.array([1.0])
    cuts = np.sort(rng.uniform(size=n - 1))
    return np.diff(np.concatenate(([0.0], cuts, [1.0])))


def get_all_categories(file_content: str) -> list:
    """
    Collect every unique category mentioned by any project,
    across ALL of a project's listed categories (not just the first).
    """
    lines = file_content.splitlines()
    section = ""
    header = []
    reader = csv.reader(lines, delimiter=";")
    category_set = set()
    for row in reader:
        if len(row) == 0 or (len(row) == 1 and len(row[0].strip()) == 0):
            continue
        if str(row[0]).strip().lower() in ["meta", "projects", "votes"]:
            section = str(row[0]).strip().lower()
            header = next(reader)
        elif section == "projects":
            for i in range(len(row)):
                key = header[i].strip()
                if row[i].strip().lower() != "none" and key in ["category", "categories"]:
                    for cat in row[i].split(","):
                        cat = cat.strip()
                        if cat:
                            category_set.add(cat)
    return sorted(category_set)  # sorted for deterministic dimension ordering


def md_parse_pabulib_from_string_random_split(
    file_content: str,
    seed: int | None = None,
) -> tuple[MDInstance, Profile]:
    """
    Parse a PaBuLib file into a multidimensional instance where:
      - d = number of unique categories across ALL projects (using every
        listed category, not just the first).
      - Each project's cost is randomly split among its categories by
        sampling uniformly from the simplex.

    Parameters
    ----------
    file_content : str
        The raw text of a PaBuLib .pb file.
    seed : int | None
        Optional RNG seed for reproducibility of the random cost splits.
    """
    rng = np.random.default_rng(seed)

    instance = MDInstance()
    ballots = []
    optional_sets = {"categories": set(), "targets": set()}

    category_list = get_all_categories(file_content)
    category_count = len(category_list)
    cat_to_idx = {cat: idx for idx, cat in enumerate(category_list)}

    lines = file_content.splitlines()
    section = ""
    header = []
    reader = csv.reader(lines, delimiter=";")
    for row in reader:
        if len(row) == 0 or (len(row) == 1 and len(row[0].strip()) == 0):
            continue
        if str(row[0]).strip().lower() in ["meta", "projects", "votes"]:
            section = str(row[0]).strip().lower()
            header = next(reader)
        elif section == "meta":
            instance.meta[row[0].strip()] = row[1].strip()
        elif section == "projects":
            p = MDProject()
            project_meta = dict()
            for i in range(len(row)):
                key = header[i].strip()
                p.name = row[0].strip()
                if row[i].strip().lower() != "none":
                    if key in ["category", "categories"]:
                        # Keep ALL categories for this project
                        project_cats = {
                            entry.strip()
                            for entry in row[i].split(",")
                            if entry.strip()
                        }
                        project_meta["categories"] = project_cats
                        p.categories = set(project_cats)
                        optional_sets["categories"].update(project_cats)
                    elif key in ["target", "targets"]:
                        project_meta["targets"] = {
                            entry.strip() for entry in row[i].split(",")
                        }
                        p.targets = set(project_meta["targets"])
                        optional_sets["targets"].update(project_meta["targets"])
                    else:
                        project_meta[key] = row[i].strip()

            total_cost = str_as_frac(project_meta["cost"].replace(",", "."))
            project_cats = list(project_meta["categories"])
            n_cats = len(project_cats)

            # Sample a random split on the (n_cats - 1)-simplex
            fractions = _sample_simplex(n_cats, rng)

            cost_vec = [0 for _ in category_list]
            for j, cat in enumerate(project_cats):
                cost_vec[cat_to_idx[cat]] = total_cost * fractions[j]

            p.costs = cost_vec
            p.dimension = len(cost_vec)
            instance.add(p)
            instance.project_meta[p] = project_meta

        elif section == "votes":
            ballot_meta = dict()
            for i in range(len(row)):
                if row[i].strip().lower() != "none":
                    ballot_meta[header[i].strip()] = row[i].strip()
            vote_type = instance.meta["vote_type"]
            if vote_type in ["approval", "choose-1"]:
                ballot = ApprovalBallot()
                for project_name in ballot_meta["vote"].split(","):
                    if project_name:
                        ballot.add(instance.get_project(project_name))
                ballot_meta.pop("vote")
            elif vote_type in ["scoring", "cumulative"]:
                if vote_type == "scoring":
                    ballot = CardinalBallot()
                else:
                    ballot = CumulativeBallot()
                if "points" in ballot_meta:
                    points = ballot_meta["points"].split(",")
                    for index, project_name in enumerate(
                        ballot_meta["vote"].split(",")
                    ):
                        ballot[instance.get_project(project_name)] = str_as_frac(
                            points[index].strip()
                        )
                    ballot_meta.pop("vote")
                    ballot_meta.pop("points")
            elif vote_type == "ordinal":
                ballot = OrdinalBallot()
                for project_name in ballot_meta["vote"].split(","):
                    if project_name:
                        ballot.append(instance.get_project(project_name))
                ballot_meta.pop("vote")
            else:
                raise NotImplementedError(
                    "The PaBuLib parser cannot parse {} profiles for now.".format(
                        instance.meta["vote_type"]
                    )
                )
            ballot.meta = ballot_meta
            ballots.append(ballot)

    # Budget: split total budget evenly across dimensions
    budget_vec = [
        str_as_frac(instance.meta["budget"].replace(",", ".")) / category_count
        for _ in category_list
    ]
    instance.budget_limits = budget_vec

    # --- Legal ballot constraints (unchanged from original) ---
    legal_min_length = instance.meta.get("min_length", None)
    if legal_min_length is not None:
        legal_min_length = int(legal_min_length)
        if legal_min_length == 1:
            legal_min_length = None
    legal_max_length = instance.meta.get("max_length", None)
    if legal_max_length is not None:
        legal_max_length = int(legal_max_length)
        if legal_max_length >= len(instance):
            legal_max_length = None
    legal_min_cost = instance.meta.get("min_sum_cost", None)
    if legal_min_cost is not None:
        legal_min_cost = str_as_frac(legal_min_cost)
        if legal_min_cost == 0:
            legal_min_cost = None
    legal_max_cost = instance.meta.get("max_sum_cost", None)
    if legal_max_cost is not None:
        legal_max_cost = str_as_frac(legal_max_cost)
        if all(legal_max_cost >= instance.budget_limits[d] for d in range(category_count)):
            legal_max_cost = None
    legal_min_total_score = instance.meta.get("min_sum_points", None)
    if legal_min_total_score is not None:
        legal_min_total_score = str_as_frac(legal_min_total_score)
        if legal_min_total_score == 0:
            legal_min_total_score = None
    legal_max_total_score = instance.meta.get("max_sum_points", None)
    if legal_max_total_score is not None:
        legal_max_total_score = str_as_frac(legal_max_total_score)
    legal_min_score = instance.meta.get("min_points", None)
    if legal_min_score is not None:
        legal_min_score = str_as_frac(legal_min_score)
        if legal_min_score == 0:
            legal_min_score = None
    legal_max_score = instance.meta.get("max_points", None)
    if legal_max_score is not None:
        legal_max_score = str_as_frac(legal_max_score)
        if legal_max_score == legal_max_total_score:
            legal_max_score = None

    profile = None
    if instance.meta["vote_type"] in ["approval", "choose-1"]:
        profile = ApprovalProfile(
            deepcopy(ballots),
            legal_min_length=legal_min_length,
            legal_max_length=legal_max_length,
            legal_min_cost=legal_min_cost,
            legal_max_cost=legal_max_cost,
        )
    elif instance.meta["vote_type"] == "scoring":
        profile = CardinalProfile(
            deepcopy(ballots),
            legal_min_length=legal_min_length,
            legal_max_length=legal_max_length,
            legal_min_score=legal_min_score,
            legal_max_score=legal_max_score,
        )
    elif instance.meta["vote_type"] == "cumulative":
        profile = CumulativeProfile(
            deepcopy(ballots),
            legal_min_length=legal_min_length,
            legal_max_length=legal_max_length,
            legal_min_score=legal_min_score,
            legal_max_score=legal_max_score,
            legal_min_total_score=legal_min_total_score,
            legal_max_total_score=legal_max_total_score,
        )
    elif instance.meta["vote_type"] == "ordinal":
        profile = OrdinalProfile(
            deepcopy(ballots),
            legal_min_length=legal_min_length,
            legal_max_length=legal_max_length,
        )

    instance.categories = optional_sets["categories"]
    instance.targets = optional_sets["targets"]

    return instance, profile


def md_parse_pabulib_random_split(
    file_path: str,
    seed: int | None = None,
) -> tuple[MDInstance, Profile]:
    """
    Parses a PaBuLib file into a multidimensional instance with random
    simplex cost splits across all of each project's categories.

    Parameters
    ----------
    file_path : str
        Path to the PaBuLib file.
    seed : int | None
        Optional RNG seed for reproducibility.

    Returns
    -------
    tuple[MDInstance, Profile]
    """
    with open(file_path, "r", newline="", encoding="utf-8-sig") as csvfile:
        instance, profile = md_parse_pabulib_from_string_random_split(
            csvfile.read(), seed=seed
        )

    instance.file_path = file_path
    instance.file_name = os.path.basename(file_path)

    return instance, profile