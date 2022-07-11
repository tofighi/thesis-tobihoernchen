from cgitb import reset
from alpyne.client.alpyne_client import AlpyneClient
from alpyne.client.abstract import BaseAlpyneEnv
from alpyne.data.spaces import Configuration, Observation, Action
from alpyne.data.model_data import ModelData
import gym
from gym import spaces
from typing import Union, Tuple, Dict, Optional, NoReturn


class JobShop(BaseAlpyneEnv):
    def __init__(self, model_path, max_steps, verbose = False):
        self.client = AlpyneClient(model_path, port=51151, verbose = verbose)
        self.config = Configuration(test=1)
        self.run = self.client.create_reinforcement_learning(self.config)
        super().__init__(self.run)
        self.reset()
        self.max_steps = max_steps

    def reset(self) -> "BaseAlpyneEnv.PyObservationType":
        self.stepcounter = 0
        self.sim.reset()
        self.sim.wait_for_completion()
        alpyne_obs = self.sim.get_observation()
        obs = self._convert_from_observation(alpyne_obs)
        return obs

    def step(
        self, action: "BaseAlpyneEnv.PyActionType"
    ) -> Tuple["BaseAlpyneEnv.PyObservationType", float, bool, Optional[dict]]:
        self.stepcounter += 1
        return super().step(action)

    def _get_observation_space(self) -> spaces.Space:
        """Describe the dimensions and bounds of the observation"""
        return spaces.Box(
            low=0, high=1, shape=(len(self.sim.get_observation().values()[0]),)
        )

    def _convert_from_observation(
        self, observation: Observation
    ) -> "BaseAlpyneEnv.PyObservationType":
        """Convert your Observation object to the format expected by Gym"""
        return observation.values()[0]

    def _get_action_space(self) -> spaces.Space:
        return spaces.Box(low=0, high=4, shape=(1,))

    def _convert_to_action(self, action: "BaseAlpyneEnv.PyActionType") -> Action:
        """Convert the action sent as part of the Gym interface to an Alpyne Action object"""
        return Action(
            data=[
                {
                    "name": "_action",
                    "type": "INTEGER_ARRAY",
                    "value": [
                        action,
                    ],
                    "unit": None,
                },
            ]
        )

    def _calc_reward(self, observation: Observation) -> float:
        """Evaluate the performance of the last action based on the current observation"""
        return observation.values()[1] if self.stepcounter != self.max_steps else -10

    def _terminal_alternative(self, observation: Observation) -> bool:
        """Optional method to add *extra* terminating conditions"""
        return self.stepcounter >= self.max_steps
