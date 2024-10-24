from SDPModel import SDPModel
from SDPPolicy import SDPPolicy
from copy import deepcopy, copy


class DummyModel(SDPModel):
    def exog_info_fn(self, decision):
        return {"W": self.prng.random()}

    def transition_fn(self, decision, exog_info):
        return {"S": exog_info["W"]}

    def objective_fn(self, decision, exog_info):
        return 0.0


class DummyPolicy(SDPPolicy):
    def get_decision(self, state):
        return {"x": 0}


model = DummyModel(state_names=["S"], decision_names=["x"], S0={"S": 0.0}, T=10)
# Initialize different policies (different thetas) with a deep copy of the model to guarantee
# that both are run with the same random values from the prng.
policy = DummyPolicy(model=deepcopy(model), policy_name="dummy policy")
policy2 = DummyPolicy(model=deepcopy(model), policy_name="dummy policy2")
print(policy.policy_name)
policy.run_policy(n_iterations=3)
print(policy2.policy_name)
policy2.run_policy(n_iterations=2)
