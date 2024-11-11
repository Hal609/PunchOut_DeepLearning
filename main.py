from PunchOutModel import PunchOutModel
import PunchOutPolicies as pop
from utils import *

model = PunchOutModel(headless=False, show_debug=True)

deepq_policy = pop.DQNAgent(model=model)

deepq_policy.train_agent(n_episodes=500)

# print(f"Random policy: {random_policy.run_policy(5)}")
# print(f"Uppercut policy: {uppercut_policy.run_policy(5)}")
# print(f"Deep Q policy: {deepq_policy.run_policy(5)}")