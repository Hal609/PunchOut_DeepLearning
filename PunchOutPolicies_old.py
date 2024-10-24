import sys
import random

sys.path.append("../")
from BaseClasses.SDPModel import SDPModel
from BaseClasses.SDPPolicy import SDPPolicy


class RandomChoice(SDPPolicy):
    def __init__(self, model: SDPModel, policy_name: str = "RadomChoice"):
        super().__init__(model, policy_name)
        

    def get_decision(self, state, t, T):
        inputs = {"RIGHT": 0, "LEFT": 0,"DOWN": 0, "UP": 0, "START": 0, "SELECT": 0, "B": 0, "A": 0}

        random_int = random.randint(0, 255)

        inputs = {key: (random_int >> i) & 1 for i, key in enumerate(inputs)}

        return inputs
    
class RandomPatient(SDPPolicy):
    def __init__(self, model: SDPModel, policy_name: str = "RadomChoice"):
        super().__init__(model, policy_name)

    def get_decision(self, state, t, T):
        inputs = {"RIGHT": 0, "LEFT": 0,"DOWN": 0, "UP": 0, "START": 0, "SELECT": 0, "B": 0, "A": 0}

        if self.model.state.Hearts_10s_place == 0\
              and self.model.state.Hearts_1s_place <= 3\
              and self.model.fight_start is not None \
                  and not self.model.state.Clock_Stop_Flag:
            random_num = random.random() 
            if random_num < 0.05:
                 inputs["UP"] = 1
                 inputs["B"] = 1
            elif random_num < 0.3:
                inputs["RIGHT"] = 1
            elif random_num < 0.6:
                inputs["LEFT"] = 1
            return inputs

        inputs = {key: (random.randint(0, 255) >> i) & 1 for i, key in enumerate(inputs)}

        return inputs
    
class SpamUppercut(SDPPolicy):
    def __init__(self, model: SDPModel, policy_name: str = "RadomChoice"):
        super().__init__(model, policy_name)

    def get_decision(self, state, t, T):
        if t & 2 == 0: return {"RIGHT": 0, "LEFT": 0,"DOWN": 0, "UP": 1, "START": 1, "SELECT": 0, "B": 0, "A": 1}
        else: return {"RIGHT": 0, "LEFT": 0,"DOWN": 0, "UP": 1, "START": 0, "SELECT": 0, "B": 0, "A": 0}
       

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

# Define the Q-Network
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_observations, 128)  # Input is the set of ram values
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)  # Output is the Q-value for each action (RIGHT, LEFT, DOWN, UP, START, SELECT", B, A)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Agent class for DQN
class DQNAgent(SDPPolicy):
    def __init__(self, model: SDPModel, policy_name: str = "Deep Q Agent"):
        super().__init__(model, policy_name)

        # Get the number of state observations
        n_observations = len(model.state_names)
        n_actions = 2**len(model.decision_names) # 2 to the power of num buttons to capture all combos of buttons

        self.qmodel = DQN(n_observations, n_actions)
        self.target_qmodel = DQN(n_observations, n_actions)  # Target network for stable training
        self.target_qmodel.load_state_dict(self.qmodel.state_dict())
        self.optimizer = optim.Adam(self.qmodel.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.update_target_every = 10
        self.steps = 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_decision(self, state):
        '''
        RUN ONCE TRAINED.
        Returns the decision from the trained model.
        '''
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        best_action = torch.argmax(q_values).item()  # Choose action with highest Q-value (exploit)
        action_name = ["RIGHT", "LEFT", "DOWN", "UP", "START", "SELECT", "B", "A"][best_action]
        action_dict = {"RIGHT": 0, "LEFT": 0,"DOWN": 0, "UP": 0, "START": 0, "SELECT": 0, "B": 0, "A": 0}
        action_dict[action_name] = 1

        return action_dict
    
    def act(self, state):
        '''
        Returns an integer representing multiple simultaneous inputs based on Q-values.
        '''
        if np.random.rand() < self.epsilon:  # Exploration: return a random integer between 0 and 255
            return random.randint(0, 255)

        # Exploitation: Use Q-values to make decisions
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.qmodel(state)
        return torch.argmax(q_values).item()  # Choose action with highest Q-value (exploit)

    def action_dict_to_num(self, action_dict):
        # Encoding the action dictionary into a single integer
        # Create a list of the action values in a specific order
        action_list = [action_dict["RIGHT"], action_dict["LEFT"], action_dict["DOWN"], 
                    action_dict["UP"], action_dict["START"], action_dict["SELECT"], 
                    action_dict["B"], action_dict["A"]]
        
        # Convert the list of binary values into an integer
        action_int = sum([val << i for i, val in enumerate(action_list)])
        return action_int

    # Decoding the integer back into a dictionary format
    def action_num_to_dict(self, action_num):
        action_list = [(action_num >> i) & 1 for i in range(8)]
        action_dict = {
            "RIGHT": action_list[0], "LEFT": action_list[1], "DOWN": action_list[2], "UP": action_list[3],
            "START": action_list[4], "SELECT": action_list[5], "B": action_list[6], "A": action_list[7]
        }
        return action_dict

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        self.model.reset(reset_prng=False) # Reload save state in emulator

        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            target = self.qmodel(state)
            with torch.no_grad():
                target_next = self.target_qmodel(next_state)

            target[0][action] = reward + (self.gamma * torch.max(target_next)) * (1 - done)

            self.optimizer.zero_grad()
            loss = self.loss_fn(self.qmodel(state), target)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_qmodel.load_state_dict(self.qmodel.state_dict())

    def train_agent(self):
        '''
        Main training function for model
        '''
        # Training loop

        n_episodes = 20

        for episode in range(n_episodes):
            done = False
            total_reward = 0

            while not done:
                action = self.act(self.model.state)
                action_dict = self.action_num_to_dict(action)
                exog = self.model.exog_info_fn(action_dict)
                next_state = self.model.step(action_dict)
                reward = self.model.objective_fn(next_state, exog)
                done = self.model.is_finished()
                
                self.remember(self.model.state, action, reward, next_state, done)
                self.model.state = next_state
                total_reward += reward

                if done: break

            self.replay()
            if episode % self.update_target_every == 0:
                self.update_target_network()
    
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