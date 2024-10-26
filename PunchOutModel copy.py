import sys

sys.path.append("../")
from BaseClasses.SDPModel import SDPModel
from runner_mac import NESWindow
from utils import *

class PunchOutModel(SDPModel):
    def __init__(
        self,
        t0: float = 0, # t0 (float, optional): Initial time. Defaults to 0.
        T: float = 0, # T (float, optional): Terminal time. Defaults to 9999.
        seed: int = 42, # seed (int, optional): Seed for random number generation. Defaults to 42.
        headless=False,
        show_debug=True
    ):
        
        state_names = state_list_from_csv("ram_data.csv")
        decision_names = ["RIGHT", "LEFT", "DOWN", "UP", "START", "SELECT", "B", "A"]

        self.game = NESWindow(rom_path="punch.nes", font_path="droid_mono.ttf", headless=headless, show_debug=show_debug)
        self.game.setup()

        self.fight_start = None
        self.ram_dict = ram_dict_from_csv("ram_data.csv")

        S0 = {}
        for key in state_names:
            S0[key] = 0

        self.total_reward = 0

        super().__init__(state_names, decision_names, S0, T, seed)
    
    def is_finished(self):
        """
        Check if the model is finished.
        Returns True if the run is finished, False otherwise.
        """
        if self.fight_start is not None:
            if self.state.Mac_Losses > 0: return True
            if self.state.Mac_Knocked_Down_Count > 0: return True
            if self.state.Current_Round > 1: return True
        return False
            
    def exog_info_fn(self, decision):
        # Return new RAM values
        new_values = {}
        for key in self.ram_dict.keys():
            new_values[key] = self.game.nes[self.ram_dict[key]]
        return new_values
    
    def transition_fn(self, decision):
        exog_info = self.exog_info_fn(decision)
        new_state = self.build_state(exog_info)

        self.check_and_save(new_state)

        return new_state
    
    def check_and_save(self, state):
        if state.Global_Variable_for_Enemy_Actions == 115 and self.fight_start is None:
            self.fight_start = self.game.nes.save()

    def objective_fn(self, decision, exog_info):
        print(process_frame_buffer(self.game.frame))
        value = 0
        if self.fight_start is None or exog_info["Fight_Started_1_fight_started_0_between_rounds"] == 0: return value

        # Reward deliberate play i.e. not spamming the controller.
        input_count = str(decision.values()).count("1")
        if input_count <= 2:
            value += 0.05

        value += 1000 * (exog_info["Opponent_ID"] > self.state.Opponent_ID)
        if exog_info["Clock_Seconds"] > self.state.Clock_Seconds:
            value -= 0.1
        if exog_info["Tens_Digit_of_Score"] != self.state.Tens_Digit_of_Score:
            value += 30
        if exog_info["Hundreds_Digit_of_Score"] > self.state.Hundreds_Digit_of_Score:
            value += 100
        if exog_info["Thousands_Digit_of_Score"] > self.state.Thousands_Digit_of_Score:
            value += 200
        if exog_info["Hearts_1s_place"] < self.state.Hearts_1s_place:
            value -= 2
        health_change = exog_info["Macs_Health"] - self.state.Macs_Health
        if health_change < 0:
            value += health_change

        self.total_reward += value

        if abs(value) > 0.1:
            self.game.reward_view_string = f"\nReward = {round(value, 2)}, Total reward = {int(self.total_reward)}"
        return value
    
    def reset(self, reset_prng: bool = False):
        """
        Resets the SDPModel to its initial state.

        This method resets the state, objective, and time variables of the SDPModel
        to their initial values.

        Parameters:
            None

        Returns:
            None
        """
        if self.fight_start is not None:
            self.game.nes.load(self.fight_start)

        self.total_reward = 0
        self.state = self.initial_state
        self.objective = 0.0
        self.t = self.t0

    def step_emu(self, decision):
        self.game.inputs = decision
        self.game.step()

    def step(self, decision):
        """
        Performs a single step in the sequential decision problem.

        Args:
            decision (namedtuple): The decision made at the current state.

        Returns:
            The new state after the step and a flag indicating if the episode is finished.
        """
        self.step_emu(decision)

        new_state = self.transition_fn(decision)

        return new_state
    
        # # Generate new exogenous information W_t+1
        exog_info = self.exog_info_fn(decision)

        # # Compute objective C_t based on W_t+1, x_t, S_t (state is not updated yet)
        # self.objective += self.objective_fn(decision, exog_info)

        # # Execute transition function and add new state to exog_info dict
        # exog_info.update(self.transition_fn(decision, exog_info))
        
        # Build new state from state variables and (optionally) exog_info variables.
        # This is convenient if some of the exogenous variables are also state variables.
        self.state = self.build_state(exog_info)

        # Update time counter
        self.update_t()

        # From the returned state S_t+1, the policy generates a new decision
        return self.state