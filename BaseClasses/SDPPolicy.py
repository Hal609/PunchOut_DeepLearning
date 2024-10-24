from copy import copy
from abc import ABC, abstractmethod
import pandas as pd
from . import SDPModel
import sys
from pretty_progress import progress_bar


class SDPPolicy(ABC):
    '''
    Inherits from ABC

    Abstract class for a sequential decision problem policy

    Attributes:
        model (SDPModel): Sequential decision problem model object containing the states and decisions as well as the transition, objective and exogenous information functions.
        policy_name (string, optional): Name of the policy
        results (DataFrame): DataFrame to store the results of each step of every iteration of the policy being run
        performance (scalar): Value to determine the performance of the policy

    Methods:
        get_decision: Abstract method to be implemented by specific policy
        run_policy: Iteratively runs the policy on the model up to the model's final timestep T. Once all iteration are complete returns the mean performance 
    '''

    def __init__(self, model: SDPModel, policy_name: str = ""):
        self.model = model
        self.policy_name = policy_name
        self.results = pd.DataFrame()
        self.performance = pd.NA

    @abstractmethod
    def get_decision(self, state, t, T):
        """
        Returns the decision made by the policy based on the given state.

        Args:
            state (namedtuple): The current state of the system.
            t (float): The current time step.
            T (float): The end of the time horizon / total number of time steps.

        Returns:
            dict: The decision made by the policy.
        """
        pass
    

    def run_policy(self, n_iterations: int = 1):
        """
        Runs the policy over the time horizon [0,T] for a specified number of iterations and return the mean performance.

        Args:
            n_iterations (int): The number of iterations to run the policy. Default is 1.

        Returns:
            None
        """
        result_list = []
        # Note: the random number generator is not reset when calling copy().
        # When calling deepcopy(), it is reset (then all iterations are exactly the same).
        for i in range(n_iterations):
            progress_bar(i, n_iterations)
            model_copy = (self.model)
            model_copy.episode_counter = i
            model_copy.reset(reset_prng=False)
            state_t_plus_1 = None
            while model_copy.is_finished() is False:
                state_t = model_copy.state
                decision_t = model_copy.build_decision(self.get_decision(state_t, model_copy.t, model_copy.T))

                # Logging
                results_dict = {"N": i, "t": model_copy.t, "C_t sum": model_copy.objective}
                results_dict.update(state_t._asdict())
                results_dict.update(decision_t._asdict())
                result_list.append(results_dict)

                state_t_plus_1 = model_copy.step(decision_t._asdict())

            results_dict = {"N": i, "t": model_copy.t, "C_t sum": model_copy.objective}
            if state_t_plus_1 is not None:
                results_dict.update(state_t_plus_1._asdict())
            result_list.append(results_dict)

        # Logging
        self.results = pd.DataFrame.from_dict(result_list)
        # t_end per iteration
        self.results["t_end"] = self.results.groupby("N")["t"].transform("max")

        # performance of one iteration is the cumulative objective at t_end
        self.performance = self.results.loc[self.results["t"] == self.results["t_end"], ["N", "C_t sum"]]
        self.performance = self.performance.set_index("N")

        # For reporting, convert cumulative objective to contribution per time
        self.results["C_t"] = self.results.groupby("N")["C_t sum"].diff().shift(-1)

        if self.results["C_t sum"].isna().sum() > 0:
            print(f"Warning! For {self.results['C_t sum'].isna().sum()} iterations the performance was NaN.")

        return self.performance.mean().iloc[0]

def show_progress(i, total_length):
    bar_length = 50     # Length of the progress bar (10 characters, one for each 10%)

    # Calculate percentage of completion
    percent_complete = (i + 1) / total_length
    num_x = int(percent_complete * bar_length)  # Number of 'x' to display

    # Build the progress bar string
    bar = '|' + '#' * num_x + '-' * (bar_length - num_x) + '|'

    # Print the progress bar (overwrite previous line)
    sys.stdout.write(f'\r{bar} {int(percent_complete * 100)}%')
    sys.stdout.flush()

    if i == total_length - 1:
        print("\n Dome")