import gurobipy as gp
from gurobipy import GRB
import sys
import random

# note: add this path properly and remove this.
sys.path.append('/Users/bencookson/Documents/Shah/PB/pabutools_MD')

from pabutools.rules.md_mes.md_mes_rule import *
from pabutools.election import *
from pabutools.election.satisfaction.md_satisfaction import *
from pabutools.election.satisfaction.satisfactionprofile import *

# --- 1. Helper Function to Check for Infinite Deviation (Zero -> Positive Utility) ---
def check_infinite_deviation(votes, costs, budget, current_utilities, sat_values):
    """
    Checks if there exists a coalition S of voters who ALL have 0 current utility,
    but can afford a bundle T with positive utility.
    """
    zero_util_voters = [i for i, u in enumerate(current_utilities) if u == 0]
    
    # If no one has 0 utility, infinite deviation is impossible.
    if not zero_util_voters:
        return False, [], []

    num_voters = len(votes)
    num_projects = len(costs)
    dimension = len(costs[0])
    
    m = gp.Model("Zero_Check")
    m.setParam('OutputFlag', 0)

    # Variables
    y = m.addVars(num_projects, vtype=GRB.BINARY, name="y")
    x = m.addVars(num_voters, vtype=GRB.BINARY, name="x")

    # 1. Force S to only contain zero-utility voters
    for i in range(num_voters):
        if i not in zero_util_voters:
            m.addConstr(x[i] == 0)

    # 2. Budget Constraint
    # Sum of costs <= Share of Budget
    for d in range(dimension):
        m.addConstr(
            (num_voters * gp.quicksum(costs[j][d] * y[j] for j in range(num_projects))) <= 
            (budget * gp.quicksum(x[i] for i in range(num_voters)))
        )

    # 3. Approval Constraint
    M_prime = 3 
    for j in range(num_projects):
        for i in zero_util_voters:
            m.addConstr(votes[i][j] + M_prime*(2 - x[i] - y[j]) >= 1)

    # 4. Non-Empty S
    m.addConstr(gp.quicksum(x[i] for i in zero_util_voters) >= 1)

    # 5. Positive Utility Constraint
    m.addConstr(gp.quicksum(y[j] * sat_values[j] for j in range(num_projects)) >= 1)

    # --- FIX: Set Objective to 0 for Feasibility Check ---
    m.setObjective(0, GRB.MINIMIZE)
    m.optimize()

    if m.Status == GRB.OPTIMAL:
        return True, [y[j].X for j in range(num_projects)], [x[i].X for i in range(num_voters)]
    
    return False, [], []


# --- 2. Main EJR Calculation Function ---
def calculate_ejr_approximation_approval(votes, costs, budget, outcome_set, sat_values=None):
    num_voters = len(votes)
    num_projects = len(costs)
    dimension = len(costs[0])

    if sat_values is None:
        sat_values = [1 for _ in range(num_projects)]

    current_utilities = [
        sum(votes[i][j]*outcome_set[j]*sat_values[j] for j in range(num_projects)) 
        for i in range(num_voters)
    ]

    # --- STEP 1: Check for Infinite Deviation (Zero -> Positive) ---
    # This captures the case where alpha can be infinite.
    is_infinite, inf_T, inf_S = check_infinite_deviation(votes, costs, budget, current_utilities, sat_values)
    if is_infinite:
        return 9999.0, inf_T, inf_S

    # --- STEP 2: Standard Alpha Optimization ---
    max_sat_sum = sum(sat_values)
    M = max_sat_sum * 1000 
    littleM = 0.001

    m = gp.Model("EJR_Approximation")
    m.setParam('OutputFlag', 0)
    m.setParam('TimeLimit', 60)

    y = m.addVars(num_projects, vtype=GRB.BINARY, name="y")
    x = m.addVars(num_voters, vtype=GRB.BINARY, name="x")
    alpha = m.addVar(lb=1.0, vtype=GRB.CONTINUOUS, name="alpha")

    # 1. Budget
    for d in range(dimension):
        m.addConstr(
            (num_voters * gp.quicksum(costs[j][d] * y[j] for j in range(num_projects))) <= 
            (budget * gp.quicksum(x[i] for i in range(num_voters)))
        )

    # 2. Approval
    for j in range(num_projects):
        for i in range(num_voters):
            m.addConstr(votes[i][j] + M*(2-x[i]-y[j]) >= 1)
    
    # 3. Utility Improvement
    for i in range(num_voters):
        new_util = gp.quicksum(y[j]*sat_values[j] for j in range(num_projects))
        
        if current_utilities[i] > 0:
            # Standard constraint
            m.addConstr(alpha * current_utilities[i] + littleM - M*(1-x[i]) <= new_util)

    m.addConstr(gp.quicksum(x[i] for i in range(num_voters)) >= 1)
    m.addConstr(gp.quicksum(y[j] for j in range(num_projects)) >= 1)

    m.setObjective(alpha, GRB.MAXIMIZE)
    m.optimize()

    # --- UPDATED RETURN LOGIC ---
    if m.Status == GRB.OPTIMAL:
        return alpha.X, [y[j].X for j in range(num_projects)], [x[i].X for i in range(num_voters)]
    
    elif m.Status == GRB.INF_OR_UNBD or m.Status == GRB.INFEASIBLE:
        # Because of step 1, this status MUST mean INFEASIBLE.
        # Infeasibility means no coalition exists that can improve upon the status quo.
        # This is a Perfect Result (Approx = 1.0).
        return 1.0, [], []
        
    else:
        print(f"Solver exited with unexpected status {m.Status}")
        return 1.0, [], []

def generate_cost_vector(dimension, cost_lb, cost_ub):
    """
    Generates a random cost vector for a multi-dimensional project.
    Ensures that the project is not 'free' (sum of costs > 0).
    """
    cost_vector = [random.randint(cost_lb, cost_ub) for _ in range(dimension)]
    while sum(cost_vector) == 0:
        cost_vector = [random.randint(cost_lb, cost_ub) for _ in range(dimension)]
    return cost_vector

def generate_voter_profile(num_projects, approval_percentage):
    """
    Generates a binary approval vector for a single voter.
    'approval_percentage' is the probability a voter approves a specific project.
    """
    voter_profile = [1 if random.random() <= approval_percentage else 0 for _ in range(num_projects)]
    while sum(voter_profile) == 0:
        voter_profile = [1 if random.random() <= approval_percentage else 0 for _ in range(num_projects)]
    return voter_profile

class RandomAdditiveSatGenerator:
    """
    A 'Functor' class used to inject dynamic, trial-specific satisfaction values
    into the AdditiveSatisfaction class.
    
    Why this is needed: The AdditiveSatisfaction constructor expects a function 
    with a specific signature, but we need that function to know about the 
    random values generated for *this specific trial*.
    """
    def __init__(self, project_values: dict[Project, Numeric]):
        # Store the dictionary mapping Project objects -> Satisfaction Scores
        self.project_values = project_values

    def __call__(self, 
                 instance: Instance, 
                 profile: AbstractProfile, 
                 ballot: AbstractBallot, 
                 project: Project, 
                 precomputed_values: dict) -> Numeric:
        # If the voter approved the project (it's in their ballot), return its assigned score.
        if project in ballot:
            return self.project_values.get(project, 0)
        return 0

if __name__ == "__main__":
    # --- Simulation Parameters ---
    dim = 5
    trials = 100
    ave_approx = 0

    num_voters = 5
    num_projects = 250
    personal_budget = 10
    budget = personal_budget * num_voters 
    cost_range = [0, 25]
    approval_percentage = 0.9  # x% chance to approve any given project

    for t in range(trials):
        # 1. Generate Raw Data (Vectors of ints)
        v_votes = [generate_voter_profile(num_projects, approval_percentage) for _ in range(num_voters)]
        v_costs = [generate_cost_vector(dim, cost_range[0], cost_range[1]) for _ in range(num_projects)]
        v_sat_scores = [random.randint(1, 5) for _ in range(num_projects)]

        # 2. Create Pabutools Objects
        # Convert raw costs into MDProject objects
        project_list = [MDProject(f"p{j}", dim, v_costs[j]) for j in range(num_projects)]

        # Map the Project objects to this trial's random satisfaction scores
        project_sat_dict = {project_list[j]: v_sat_scores[j] for j in range(num_projects)}
        
        # Instantiate our custom satisfaction generator with this trial's data
        sat_func = RandomAdditiveSatGenerator(project_sat_dict)

        # Create the Instance and populate it
        instance = MDInstance()
        for x in project_list:
            instance.add(x)
        instance.budget_limits = [budget for _ in range(dim)]
        
        # 3. Create Ballots and Profile
        ballot_list = []
        for i in range(num_voters):
            i_approvals = [p for bit, p in zip(v_votes[i], project_list) if bit == 1]
            ballot_list.append(ApprovalBallot(i_approvals))
        profile = ApprovalProfile(ballot_list)

        # 4. Construct Custom Satisfaction Profile
        # We manually create the list of AdditiveSatisfaction objects here.
        # This is the only way to pass our custom 'sat_func' (the evaluator) into them.
        sat_measures_list = [
            AdditiveSatisfaction(
                instance=instance, 
                profile=profile, 
                ballot=ballot, 
                func=sat_func  # Injecting the functor
            )
            for ballot in profile
        ]

        # Initialize SatisfactionProfile with the pre-built list.
        sat_profile = SatisfactionProfile(init=sat_measures_list, instance=instance)
        sat_profile.sat_class = AdditiveSatisfaction
        
        # 5. Run the Voting Rule (Method of Equal Shares - Multi-Dimensional)
        outcome = naive_md_mes(
            instance, 
            profile, 
            Cardinality_Sat, 
            [personal_budget for _ in range(dim)], 
            projection_function_max, 
            sat_profile
        )
        
        # 6. Convert Outcome to Binary Vector for Analysis
        v_outcome = []
        for j in range(num_projects):
            if project_list[j] in outcome:
                v_outcome.append(1)
            else:
                v_outcome.append(0)
        
        # 7. Calculate EJR Approximation (Using the ILP defined previously)
        approx, T, S = calculate_ejr_approximation_approval(
            v_votes, v_costs, budget, v_outcome, sat_values=v_sat_scores
        )
        
        print(f"Trial {t}: Approx = {approx}")
        ave_approx += approx
        
        # Break if we hit the "Infinite Deviation" flag (9999)
        if approx > 9000:
            print("UNBOUNDED ERROR: BREAKING")
            break
    
    # 8. Print Average Approximation Ratio
    print(f"Average Approximation: {ave_approx/trials}")
        
    # --- Manual Test Case ---
    # print("\n--- Running Manual Test Case ---")
    # v_votes = [[1,1,1,1,1,0,0,0,0,0],[0,0,0,0,0,1,1,1,1,1]]
    # # Note: v_costs has dimensions 10x2 (10 projects, 2 dims)
    # v_costs = [[8,2],[9,4],[9,7],[4,7],[8,1],[0,5],[2,5],[3,3],[7,2],[3,3]]
    # # Total budget for the election
    # v_budget = 20
    # v_outcome = [1,0,0,0,0,0,0,0,0,1] # Projects 7 and 9 selected
    # v_sat_scores = [5,6,2,9,1,7,1,12,4,10]

    # approx, T, S = calculate_ejr_approximation_approval(v_votes, v_costs, v_budget, v_outcome, v_sat_scores)
    
    # # Note: If approx > 1, the rule Failed EJR by factor approx.
    # # If you want the approx ratio (how much of the optimal utility did they get?), it's 1/approx.
    # print(f"Max Improvement (Alpha): {approx}")
    # print(f"Inverse Ratio (Optimization): {1/approx:.4f}" if approx != 0 else "0")
    # print(f"Deviating Voters (Indices): {[i for i, val in enumerate(S) if val > 0.5]}")
    # print(f"Deviating Projects (Indices): {[j for j, val in enumerate(T) if val > 0.5]}")