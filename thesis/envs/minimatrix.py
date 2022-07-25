from alpyne.client.alpyne_client import AlpyneClient
from alpyne.client.abstract import BaseAlpyneEnv
from alpyne.data.spaces import Configuration, Observation, Action
from gym import spaces
from typing import Union, Tuple, Dict, Optional, NoReturn
import torch
import random
from .randdispatcher import RandDispatcher
import re
from .build_config import build_config


class MiniMatrix(BaseAlpyneEnv):
    def __init__(
        self,
        model_path=None,
        max_steps=None,
        max_seconds=None,
        verbose=False,
        fleetsize=1,
        config_args=dict(),
        max_fleetsize=None,
        dispatcher=None,
        port=51150,
    ):
        self.fleetsize = fleetsize
        self.max_fleetsize = max_fleetsize
        self.max_steps = max_steps
        self.max_seconds = max_seconds
        self.stepcounter = 0

        if max_fleetsize is not None:
            self.shufflerule = random.sample(
                list(range(self.max_fleetsize)), self.fleetsize
            )

        if dispatcher is None:
            self.dispatcher = RandDispatcher(
                fleetsize if max_fleetsize is None else max_fleetsize
            )
        else:
            self.dispatcher = dispatcher

        if (
            model_path is not None
        ):  # can be build without env, i.e. for running in pypeline
            self.client = AlpyneClient(model_path, port=port, verbose=verbose)

            self.config = build_config(config_args, fleetsize)

            self.run = self.client.create_reinforcement_learning(self.config)
            super().__init__(self.run)
            self.reset()

    def reset(self) -> "BaseAlpyneEnv.PyObservationType":
        if self.max_fleetsize is not None:
            self.shufflerule = random.sample(
                list(range(self.max_fleetsize)), self.fleetsize
            )
        self.stepcounter = 0
        self.sim.reset()
        self.sim.wait_for_completion()
        alpyne_obs = self._catch_dispatcher(self.sim.get_observation())
        obs = self._convert_from_observation(alpyne_obs)

        return obs

    def step(
        self, action: "BaseAlpyneEnv.PyActionType"
    ) -> Tuple["BaseAlpyneEnv.PyObservationType", float, bool, Optional[dict]]:
        self.stepcounter += 1

        if self.stepcounter % 100 == 0 and self.max_fleetsize is not None:
            self.shufflerule = random.sample(
                list(range(self.max_fleetsize)), self.fleetsize
            )

        alpyne_action = self._convert_to_action(action)
        self.sim.take_action(alpyne_action)

        self.sim.wait_for_completion()

        alpyne_obs = self._catch_dispatcher(self.sim.get_observation())
        obs = self._convert_from_observation(alpyne_obs)
        reward = self._calc_reward(alpyne_obs)
        done = self.sim.is_terminal() or self._terminal_alternative(alpyne_obs)
        info = self.sim.last_state[1]  # dictionary of info

        return obs, reward, done, info

    def _catch_dispatcher(self, alpyne_obs) -> Observation:
        while alpyne_obs.caller == "Dispatching":
            action = self.dispatcher(alpyne_obs)
            self.sim.take_action(action)
            self.sim.wait_for_completion()
            alpyne_obs = self.sim.get_observation()
        return alpyne_obs

    def _get_observation_space(self) -> spaces.Box:
        """Describe the dimensions and bounds of the observation"""

        obs_sample = self._catch_dispatcher(self.sim.get_observation()).obs
        if self.max_fleetsize is None:
            shape = (torch.prod(torch.tensor(obs_sample).shape),)
        else:
            nStations = len(obs_sample) - self.fleetsize
            shape = ((self.max_fleetsize + nStations) * len(obs_sample[0]),)

        return spaces.Box(low=0, high=1, shape=shape)

    def _convert_from_observation(
        self, observation: Observation
    ) -> "BaseAlpyneEnv.PyObservationType":
        """Convert your Observation object to the format expected by Gym"""
        self.mask = self._get_mask(observation.caller, len(observation.obs))
        return self._modify_observation(observation.obs)

    def _modify_observation(self, observation: Observation) -> torch.Tensor:
        if self.max_fleetsize is None:
            obs_out = torch.Tensor(observation)
        else:
            n_obs = len(observation) - self.fleetsize + self.max_fleetsize
            obs_len = len(observation[0])
            obs_out = torch.zeros((n_obs, obs_len))
            for i in range(self.max_fleetsize):
                obs_out[i] = (
                    torch.tensor(observation[self.shufflerule.index(i)])
                    if i in self.shufflerule
                    else torch.zeros((obs_len,))
                )
            obs_out[self.max_fleetsize :] = torch.tensor(observation)[self.fleetsize :]
        return obs_out.flatten()

    def _get_mask(self, caller: str, length) -> torch.Tensor:
        if caller == "Routing" or caller == "Combined":
            if self.max_fleetsize is None:
                mask = [True for i in range(length)]
            else:
                mask = [i in self.shufflerule for i in range(length)]
        else:
            assert caller.startswith("root.agvs[")
            i_agvs = map(int, re.findall("\d+", caller))
            i_agvs_shuffled = [self.shufflerule[i] for i in i_agvs]
            mask = [i in i_agvs_shuffled for i in range(length)]
        return torch.Tensor(mask)

    def _get_action_space(self) -> spaces.MultiDiscrete:
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
        action = self._modify_action(action)
        return Action(
            data=[
                {
                    "name": "actions",
                    "type": "INTEGER_ARRAY",
                    "value": list(action),
                    "unit": None,
                },
                {
                    "name": "receiver",
                    "type": "INTEGER",
                    "value": 0,
                    "unit": None,
                },
            ]
        )

    def _modify_action(self, action):
        if self.max_fleetsize is not None:
            action = [
                action[self.shufflerule[i]] for i, _ in enumerate(self.shufflerule)
            ]
        return action

    def _calc_reward(self, observation: Observation) -> float:
        """Evaluate the performance of the last action based on the current observation"""
        return observation.rew  # if self.stepcounter != self.max_steps else -10

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
