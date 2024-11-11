import sys
import math
import random
import matplotlib.pyplot as plt
from utils import *

sys.path.append("../")
from BaseClasses.SDPModel import SDPModel
from BaseClasses.SDPPolicy import SDPPolicy

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import datetime
import signal
import time
import json
import os

from visualise_run import visualise_latest

import platform
#gradplus
# Define the Q-Network
class DQN(nn.Module):
    def __init__(self,n_pixel_vals, n_ram_values, n_outputs):
        super(DQN, self).__init__()
        
        # Adjusted Convolutional layers for pixel data
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.relu3 = nn.ReLU()
        
        # Flattened size after conv layers
        conv_output_size = 64 * 7 * 7  # 64 channels * 7 height * 7 width = 3136

        # Fully connected layer for RAM values
        self.fc_ram = nn.Linear(n_ram_values, 32) 
        self.relu_fc_ram = nn.ReLU()
        
        # Fully connected layers after concatenation
        self.fc_combined1 = nn.Linear(conv_output_size + 32, 6400)
        self.relu_fc1 = nn.ReLU()
        self.fc_combined2 = nn.Linear(6400, 512)
        self.relu_fc2 = nn.ReLU()
        self.fc_output = nn.Linear(512, n_outputs)

    def forward(self, ram_data, pixel_data):
        # Convolution pathway for pixel data
        x = torch.relu(self.conv1(pixel_data))
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.view(x.size(0), -1)  # Flatten (batch_size, 3136)
        
        # RAM pathway
        # ram = torch.relu(self.fc_ram(ram_data))  # (batch_size, 32)
        ram = self.fc_ram(ram_data)  # (batch_size, 32)
        ram = self.relu_fc_ram(ram)

        # Combine
        # combined = torch.cat((x, ram), dim=1)  # (batch_size, 3168)
        # combined = torch.relu(self.fc_combined1(combined))
        combined = torch.cat((x, ram), dim=1)  # (batch_size, 3168)
        combined = self.fc_combined1(combined)
        combined = self.relu_fc1(combined)
        combined = self.fc_combined2(combined)
        combined = self.relu_fc2(combined)
        output = self.fc_output(combined)
        # Optionally, apply ReLU to the output layer
        # output = self.relu_output(output)
        
        return output


# Agent class for DQN
class DQNAgent(SDPPolicy):
    def __init__(self, model: SDPModel, policy_name: str = "Deep Q Agent"):
        super().__init__(model, policy_name)

        self.device = global_device
        print("Using device:", self.device)

        # self.actions = compute_actions(max_simultaneous_buttons=2)
        # a b select start up down left right
        self.actions = ["00000000", "01000000", "00010000", "00001000", "00000010", "00000001", "01001000"]

        # Get the number of actions and observations
        n_ram_vals = len(model.ram_state_names)
        n_pixel_vals = len(model.pixel_state_names)
        n_actions = len(self.actions)

        self.policy_net = DQN(n_pixel_vals, n_ram_vals, n_actions).to(self.device)
        self.target_net = DQN(n_pixel_vals, n_ram_vals, n_actions).to(self.device)  # Target network for stable training
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.learning_rate = 0.00025
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.01)
        if platform.system() == "Darwin":
            self.max_memory_size = 8000
        else:
            self.max_memory_size = 100000
        self.memory = [None] * self.max_memory_size
        self.memory_index = 0
        self.memory_full = False
        self.gamma = 0.9  # Discount factor
        self.epsilon = 1.0 # Exploration rate
        self.epsilon_decay = 0.99999975
        self.epsilon_min = 0.1
        self.batch_size = 32
        self.update_target_every = 1e4
        self.steps_done = 0

        self.frame_data_file = None
        self.episode_data_file = None
        self.folder_name = None

        # Register cleanup function for common exit signals
        signal.signal(signal.SIGINT, self.exit)
        signal.signal(signal.SIGTERM, self.exit)

    def exit(self, signum=None, frame=None):
        self.model.game.clean_up(signum, frame)
        self.episode_data_file.close()
        self.frame_data_file.close()
        time.sleep(10)
        visualise_latest()

    def display_memory_progress(self):
        memory_percent = self.memory_index/self.max_memory_size * 100
        if memory_percent % 0.1 < 0.001:
            self.model.game.debug_print(f"Memory, {round(self.memory_index/self.max_memory_size * 100, 1)}% full", clear_type="self", prepend=True)

    def remember(self, ram_state, pixel_state, action_index, reward, next_ram_state, next_pixel_state, done):
        self.memory[self.memory_index] = ( ram_state, pixel_state, action_index, reward, next_ram_state, next_pixel_state, done)

        self.memory_index = (self.memory_index + 1) % self.max_memory_size
        self.display_memory_progress()
        if self.memory_index == 0:
            self.memory_full = True

    def get_decision(self, state):
        '''
        RUN ONCE TRAINED.
        Returns the decision from the trained model.
        '''
        state = torch.tensor(state, device=self.device).unsqueeze(0)
        q_values = self.model(state)
        best_action = torch.argmax(q_values).item()  # Choose action with highest Q-value (exploit)
        action_name = ["RIGHT", "LEFT", "DOWN", "UP", "START", "SELECT", "B", "A"][best_action]
        action_dict = {"RIGHT": 0, "LEFT": 0,"DOWN": 0, "UP": 0, "START": 0, "SELECT": 0, "B": 0, "A": 0}
        action_dict[action_name] = 1

        return action_dict
        
    def act(self, ram_state, pixel_state):
        '''
        Returns an integer representing multiple simultaneous inputs based on Q-values.
        a b select start up down left right
        '''
        if np.random.rand() < self.epsilon:
            choice = random.randint(0, len(self.actions) - 1)
            return choice

        # Exploitation: Use Q-values to make decisions
        with torch.no_grad():
            return self.policy_net(ram_state.unsqueeze(0), pixel_state).max(1).indices.view(1, 1)

    def replay(self):
        n_step = 10 # Number of frames for multi-step return

        memory_size = self.max_memory_size if self.memory_full else self.memory_index

        if memory_size < self.batch_size:
            return 0.0, None

        # Sample a batch with sequences of actions and states
        batch = random.sample(self.memory[:memory_size], self.batch_size)

        ram_states, pixel_states, actions, rewards, next_ram_states, next_pixel_states, dones = zip(*batch)

        # Convert to tensors and move to device
        ram_states = torch.stack(ram_states)
        pixel_states = torch.stack(pixel_states).squeeze(2)  # Remove extra dimension
        next_ram_states = torch.stack(next_ram_states)
        next_pixel_states = torch.stack(next_pixel_states).squeeze(2)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)

        # Calculate n-step returns for more stable training
        step_gamma = 0.95
        n_step_rewards = rewards.clone()  # Copy of original rewards tensor
        for i in range(1, n_step):  # For n-steps, calculate gamma-discounted reward
            n_step_rewards[:-i] += step_gamma ** (i) * rewards[i:]

        # Compute Q-values for current states
        state_action_values = self.policy_net(ram_states, pixel_states).gather(1, actions)


        next_state_values = torch.zeros(self.batch_size, device=global_device)
        with torch.no_grad():
            next_state_values = self.target_net(next_ram_states, next_pixel_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + n_step_rewards

        # # Compute the target Q-values for the next states using target_net with n-step returns
        # with torch.no_grad():
        #     next_q_values = self.target_net(next_ram_states, next_pixel_states).max(1).values
        #     # print(next_q_values)
        #     # print("next_q shape", next_q_values.shape)
        # target_q_values = n_step_rewards + (self.gamma ** n_step * next_q_values * ~dones)

        # Compute loss and optimize
        loss_func = nn.MSELoss()
        loss = loss_func(state_action_values, expected_state_action_values.unsqueeze(1))
        # loss = nn.SmoothL1Loss(q_value.squeeze(1), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return loss

    def update_target_network(self):
        TAU = 0.005
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        self.target_net.load_state_dict(target_net_state_dict)

        # self.target_net.load_state_dict(self.policy_net.state_dict())

    def create_output_files(self):
        self.folder_name = f'Outputs/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(self.folder_name)
        self.frame_data_file = open(self.folder_name + "/frame_data.csv", "a")
        self.frame_data_file.write("loss,reward,episode,frame_number,epsilon,q_vals\n")
        self.episode_data_file = open(self.folder_name + "/episode_data.csv", "a")
        self.episode_data_file.write("total_reward,episode_length,q_vals\n")
        self.episode_data_file.close()

        with open(self.folder_name + "/paramters.json", "w") as parameters_file:
            json.dump({
                "Learning Rate": self.learning_rate,
                "Optimiser": str(type(self.optimizer)),
                "Max Memory Size":self.max_memory_size,
                "Gamma": self.gamma,
                "Epsilon0": self.epsilon,
                "Epsilon Decay Factor": self.epsilon_decay,
                "Epsilon Min": self.epsilon_min,
                "Batch Size": self.batch_size,
                "Update Target Every": self.update_target_every  
            }, parameters_file, indent=4)

    def train_agent(self, n_episodes):
        '''
        Main training function for model
        '''
        self.create_output_files()

        total_rewards = []
        frame_num = 0

        # Training loop
        for episode in range(n_episodes):
            done = False
            total_reward = 0
            episode_duration = 0
            frame_num = frame_num % self.update_target_every

            while not done:
                frame_num += 1
                episode_duration += 1

                action_index = self.act(self.model.ram_state, self.model.pixel_state)
                self.model.step_emu(int(self.actions[action_index], 2))
                exog = self.model.exog_info_fn()
                next_ram_state, next_pixel_state = self.model.transition_fn(exog)
                reward = self.model.objective_fn(exog)
                done = self.model.is_finished()

                if round(self.model.current_ram["Fight_Started_1_fight_started_0_between_rounds"]) > 1 or self.model.fight_start is None:
                    self.model.step_emu(int(random.choice(self.actions), 2))
                    exog = self.model.exog_info_fn()
                    self.model.state = self.model.transition_fn(exog)

                else:
                    self.remember(self.model.ram_state, self.model.pixel_state, action_index, reward, next_ram_state, next_pixel_state, done)
                    self.model.ram_state = next_ram_state
                    self.model.pixel_state = next_pixel_state
                    total_reward += reward

                    if frame_num % 4 == 0:
                        loss = self.replay()
                        # self.frame_data_file.write(f"{round(float(loss) * 100000)},{reward},{episode},{episode_duration},{self.epsilon},{0}\n")
                        # self.model.game.debug_print(f"Loss: {round(float(loss) * 100000, 1)}", clear_type="self", prepend=True)

                    # if frame_num % self.update_target_every == 0:
                        self.update_target_network()
                        self.model.game.debug_print(f"Target updated {random.random()}", clear_type="self", prepend=True)

                    if frame_num % 1200 == 0:
                        # Update epsilon
                        if self.epsilon > self.epsilon_min:
                            self.epsilon *= self.epsilon_decay
                            self.model.game.training_epsilon = round(self.epsilon, 3)

                    if done: 
                        tot = float(total_reward)
                        total_rewards.append(tot)
                        with open(self.folder_name + "/episode_data.csv", "a") as self.episode_data_file:
                            self.episode_data_file.write(f"{total_reward},{episode_duration},{0}\n")
                        self.model.game.debug_print(f"{episode}: Total reward = {round(tot, 3)} Ave of last 10: {round(sum(total_rewards[-10:])/10, 2)}, Ave first 10: {round(sum(total_rewards[:10])/10, 2)}")
                        print(f"{episode}: Total reward = {round(tot, 3)} Ave of last 10: {round(sum(total_rewards[-10:])/10, 2)}, Ave first 10: {round(sum(total_rewards[:10])/10, 2)}")
                        break

            self.model.reset(reset_prng=False)
                    
        plt.plot(np.arange(0, len(total_rewards)), total_rewards)
        plt.show()

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