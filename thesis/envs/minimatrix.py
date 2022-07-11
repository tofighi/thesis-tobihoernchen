from alpyne.client.alpyne_client import AlpyneClient
from alpyne.client.abstract import BaseAlpyneEnv
from alpyne.data.spaces import Configuration, Observation, Action
from gym import spaces
from typing import Union, Tuple, Dict, Optional, NoReturn
import numpy as np
import random


class MiniMatrix(BaseAlpyneEnv):
    def __init__(
        self,
        model_path=None,
        max_steps=None,
        max_seconds=None,
        verbose=False,
        fleetsize=1,
        max_fleetsize=None,
        port=51150,
    ):
        self.fleetsize = fleetsize
        self.max_fleetsize = max_fleetsize
        if max_fleetsize is not None:
            self.shufflerule = random.sample(
                list(range(self.max_fleetsize)), self.fleetsize
            )
        if model_path is not None:
            self.client = AlpyneClient(model_path, port=port, verbose=verbose)
            self.config = Configuration(fleetsize=fleetsize)
            self.run = self.client.create_reinforcement_learning(self.config)
            super().__init__(self.run)
            self.reset()
            self.max_steps = max_steps
            self.max_seconds = max_seconds

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
            low=0,
            high=1,
            shape=(np.array(self.sim.get_observation().values()[0]).shape)
            if self.max_fleetsize is not None
            else (self.max_fleetsize, len(self.sim.get_observation().values()[0][0])),
        )

    def _convert_from_observation(
        self, observation: Observation
    ) -> "BaseAlpyneEnv.PyObservationType":
        """Convert your Observation object to the format expected by Gym"""
        if self.max_fleetsize is None:
            return observation.values()[0]
        else:
            obs_in = observation.values()[0]
            obs_len = len(obs_in.values()[0][0])
            obs_out = []
            for i in range(len(self.max_fleetsize)):
                obs_out.append(
                    obs_in[self.shufflerule.index(i)]
                ) if i in self.shufflerule else obs_out.append([0] * obs_len)
            return obs_out

    def _get_action_space(self) -> spaces.Space:
        return spaces.MultiDiscrete(
            [
                5
                for _ in range(
                    self.fleetsize if self.max_fleetsize is None else self.max_fleetsize
                )
            ]
        )

    def _convert_to_action(self, action: "BaseAlpyneEnv.PyActionType") -> Action:
        """Convert the action sent as part of the Gym interface to an Alpyne Action object"""
        if self.max_fleetsize is not None:
            action = [act for i, act in enumerate(action) if i in self.shufflerule]
        return Action(
            data=[
                {
                    "name": "nextnodes",
                    "type": "INTEGER_ARRAY",
                    "value": action,
                    "unit": None,
                },
            ]
        )

    def _calc_reward(self, observation: Observation) -> float:
        """Evaluate the performance of the last action based on the current observation"""
        return observation.values()[1]  # if self.stepcounter != self.max_steps else -10

    def _terminal_alternative(self, observation: Observation) -> bool:
        """Optional method to add *extra* terminating conditions"""
        terminal_max_steps = (
            (self.stepcounter >= self.max_steps)
            if self.max_steps is not None
            else False
        )
        time = 0
        if self.max_seconds is not None:
            time = self.run.get_state()[1]["model_time"]
        return terminal_max_steps or time > self.max_seconds
