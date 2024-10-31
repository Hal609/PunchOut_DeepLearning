import sys

sys.path.append("../")
from BaseClasses.SDPModel import SDPModel
from runner import NESWindow
from utils import *

import torch

class PunchOutModel(SDPModel):
    def __init__(
        self,
        t0: float = 0, # t0 (float, optional): Initial time. Defaults to 0.
        T: float = 0, # T (float, optional): Terminal time. Defaults to 9999.
        seed: int = 42, # seed (int, optional): Seed for random number generation. Defaults to 42.
        headless=False,
        show_debug=True
    ):
        
        self.state_names = state_list_from_csv("ram_data_network.csv")
        self.state_names += [f"Pixel{num}" for num in range(21*21)]
        # decision_names = ["RIGHT", "LEFT", "DOWN", "UP", "START", "SELECT", "B", "A"]

        self.current_ram = {}
        self.previous_ram = {}

        self.game = NESWindow(rom_path="punch.bin", headless=headless, show_debug=show_debug)

        self.game.setup()

        self.fight_start = None
        self.ram_dict = ram_dict_from_csv("ram_data.csv")
        self.network_ram_dict = ram_dict_from_csv("ram_data_network.csv")

        S0 = {}
        for key in self.state_names:
            S0[key] = 0

        self.total_reward = 0

        super().__init__(self.state_names, S0, T, seed)
    
    def get_state_val(self, value_string):
        return self.state[self.state_names.index(value_string)]

    def build_state(self, info: dict):
        # Extract state values in the correct order
        state_values = [(info[key]) for key in self.state_names]
        # Create tensor directly from the list of values
        next_state = torch.tensor(state_values, dtype=torch.float32, device=global_device)
        return next_state
    
    def is_finished(self):
        """
        Check if the model is finished.
        Returns True if the run is finished, False otherwise.
        """
        if self.fight_start is not None:
            if self.current_ram["Mac_Losses"] > 0: return True
            if self.current_ram["Mac_Knocked_Down_Count"] > 0: return True
            if self.current_ram["Current_Round"] > 1/255: return True
            if 0 != self.current_ram["Macs_Health"] != 96/255: return True
        return False
            
    def exog_info_fn(self, decision={}):
        # Return new RAM values
        new_ram_values = {}
        new_state_values = {}
        n = 0
        for pair in process_frame_buffer(self.game.frame):
            for pixel in pair:
                new_state_values[f"Pixel{n}"] = pixel
                n += 1

        for key in self.network_ram_dict.keys():
            new_state_values[key] = self.game.nes[self.network_ram_dict[key]]/255

        for key in self.ram_dict.keys():
            new_ram_values[key] = self.game.nes[self.ram_dict[key]]/255

        self.previous_ram, self.current_ram = self.current_ram, new_ram_values

        return new_state_values
    
    def transition_fn(self, exog_info, decision={}):
        # exog_info = self.exog_info_fn(decision)
        new_state = self.build_state(exog_info)

        self.check_and_save(new_state)

        return new_state
    
    def check_and_save(self, state):
        if state[self.state_names.index("Global_Variable_for_Enemy_Actions")] == 115/255 and self.fight_start is None:
            self.fight_start = self.game.nes.save()

    def objective_fn(self, exog_info, decision={}):
        value = 0
        if self.fight_start is None or self.current_ram["Fight_Started_1_fight_started_0_between_rounds"] == 0.0: return value

        # Reward deliberate play i.e. not spamming the controller.
        # input_count = str(decision.values()).count("1")
        # if input_count <= 2:
        #     value += 0.05
        if self.current_ram["Fight_Started_1_fight_started_0_between_rounds"] == 0:
            value -= 0.001
        value += 200 * float(self.current_ram["Opponent_ID"] > self.previous_ram["Opponent_ID"])
        
        if self.current_ram["Clock_Seconds"] > self.previous_ram["Clock_Seconds"]:
            value -= 0.01
        if self.current_ram["Tens_Digit_of_Score"] != self.previous_ram["Tens_Digit_of_Score"]:
            value += 2
        if self.current_ram["Hundreds_Digit_of_Score"] > self.previous_ram["Hundreds_Digit_of_Score"]:
            value += 20
        if self.current_ram["Thousands_Digit_of_Score"] > self.previous_ram["Thousands_Digit_of_Score"]:
            value += 200
        if self.current_ram["Hearts_1s_place"] != self.previous_ram["Hearts_1s_place"]:
            value -= 0.1
        # if exog_info["Hearts_10s_place"] > self.get_state_val("Hearts_10s_place"):
        #     value += 5
        if self.current_ram["Mac_Knocked_Down_Count"] > self.previous_ram["Mac_Knocked_Down_Count"]:
            value -= 100
        health_change = self.current_ram["Macs_Health"]*255 - self.previous_ram["Macs_Health"]*255

        if health_change < 0:
            value += float(2 * health_change)

        scaled_reward = max(min(value / 100, 1.0), -1.0)
        # value = value / 800

        self.total_reward += scaled_reward

        self.game.reward_view_string = f"\nReward = {round(scaled_reward, 10)}, Total reward = {round(self.total_reward, 3)}"

        return scaled_reward
    
    def reset(self, reset_prng: bool = False):
        """
        Resets the SDPModel to its initial state.

        This method resets the state, objective, and time variables of the SDPModel
        to their initial values.
        """
        if self.fight_start is not None:
            self.game.nes.load(self.fight_start)

        self.total_reward = 0
        self.state = self.initial_state
        self.objective = 0.0
        self.t = self.t0

    def step_emu(self, decision):
        # self.game.inputs = decision
        self.game.step(decision)

    # def step(self, decision):
    #     self.step_emu(decision)

    #     new_state = self.transition_fn(decision)

    #     return new_state