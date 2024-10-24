from PunchOutModel import PunchOutModel
import PunchOutPolicies as pop
from utils import *

model = PunchOutModel(headless=False, show_debug=True)

random_policy = pop.RandomPatient(model=model)
uppercut_policy = pop.SpamUppercut(model=model)
deepq_policy = pop.DQNAgent(model=model)

deepq_policy.train_agent(n_episodes=50)

# print(f"Random policy: {random_policy.run_policy(5)}")
# print(f"Uppercut policy: {uppercut_policy.run_policy(5)}")
# print(f"Deep Q policy: {deepq_policy.run_policy(5)}")