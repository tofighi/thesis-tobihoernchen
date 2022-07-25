from alpyne.data.spaces import Action, Observation
import random

class RandDispatcher:
    def __init__(self, n_actions) -> None:
        self.n_actions = n_actions

    def __call__(self, obs:Observation)->Action:
        action_max = obs.n_nodes-1
        actions = [random.randint(0, action_max) for i in range(self.n_actions)]
        return Action(
            data=[
                {
                    "name": "actions",
                    "type": "INTEGER_ARRAY",
                    "value": list(actions),
                    "unit": None,
                },
                {
                    "name": "receiver",
                    "type": "INTEGER",
                    "value": 1,
                    "unit": None,
                },
            ]
        )
