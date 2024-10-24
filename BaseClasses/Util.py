import pandas as pd
from itertools import product
from copy import deepcopy
from . import SDPPolicy


def grid_search(grid: dict, policy: SDPPolicy.SDPPolicy, n_iterations: int, ordered: bool = False):
    '''
    Performs a grid search to optimise parameters for a policy

    Args:
        grid (dict): A dictionary mapping named parameters to lists of values to test
        policy (SDPPolicy.SDPPolicy): The policy which is to be tuned
        n_iterations (integer): The number of iterations to perform for each policy before determining the average performance
        ordered (bool, optional): 
    '''
    if len(grid) != 2 and ordered:
        ordered = False
        print("Warning: Grid search for ordered parameters only works if there are exactly two parameters.")
    best_performance = 0.0
    best_parameters = None
    rows = []
    params = grid.keys()
    
    # Passes both sets of parameter values to product to get a list every possible combination of values
    for v in product(*grid.values()):
        if ordered:
            if v[0] >= v[1]:
                continue

        # Do a deep copy so all parameter sets get the same random numbers
        policy_copy = deepcopy(policy)

        for param, value in zip(params, v):
            setattr(policy_copy, param, value)

        performance = policy_copy.run_policy(n_iterations=n_iterations)

        row = dict(zip(params, v))
        row["performance"] = performance
        rows.append(row)
        if performance > best_performance:
            best_performance = performance
            best_parameters = dict(zip(params, v))

    return {
        "best_parameters": best_parameters,
        "best_performance": best_performance,
        "all_runs": pd.DataFrame(rows),
    }
