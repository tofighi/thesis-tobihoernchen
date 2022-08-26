from alpyne.client.alpyne_client import AlpyneClient
from alpyne.client.abstract import BaseAlpyneEnv
from alpyne.data.spaces import Configuration, Observation, Action
from gym import spaces
from typing import Union, Tuple, Dict, Optional, NoReturn
import torch
import random
from .randdispatcher import RandDispatcher
import re
from ..utils.build_config import build_config


class MatrixRoutingMAGym(BaseAlpyneEnv):
    """
    Gets called by Each AGV separately and simply presents this as the next step to the Trainer. Makes state transitions wrong, so basically useless.

    model_path: If supposed to build its own client
    client: If client is given
    max_fleetsize: if None, no shuffling is done, observations are taken as they are.
        If max_fleetsize is given (should be >= fleetsize) the obs is always
        max_fleetsize agvs big and the indices are randomly allocated (shuffled every 100 steps)
    dispatcher: If None, RandDispatcher is used
    counter: reference to a List<int>, first entry is used as port/seed and the incremented
    """

    def __init__(
        self,
        model_path: str = None,
        port=51150,
        client: AlpyneClient = None,
        max_steps: int = None,
        max_seconds: int = None,
        fleetsize: int = 1,
        max_fleetsize: int = 1,
        config_args: dict = dict(),
        dispatcher=None,
        dispatcher_args=dict(),
        counter=None,
        verbose: bool = False,
    ):
        self.fleetsize = fleetsize
        self.max_fleetsize = max_fleetsize
        self.max_steps = max_steps
        self.max_seconds = max_seconds
        self.config_args = config_args
        self.stepcounter = 0
        self.context = None  # used to catch special network context info which is send on the first step

        self.shuffle()

        if dispatcher is None:
            self.dispatcher = RandDispatcher(
                fleetsize if max_fleetsize is None else max_fleetsize, **dispatcher_args
            )
        else:
            self.dispatcher = dispatcher

        if (
            model_path is not None or client is not None
        ):  # can be build without env, i.e. for running in pypeline
            if counter is not None:
                port = int(counter[0])
                counter[0] = counter[0] + 1
            if client is None:
                self.client = AlpyneClient(model_path, port=port, verbose=verbose)
            else:
                self.client = client
            config_args.update(reward_separateAgv=True, routingOnNode=True)
            self.config = build_config(config_args, fleetsize, port)
            self.run = self.client.create_reinforcement_learning(self.config)
            super().__init__(self.run)
            self.reset()

    def shuffle(self):
        self.shufflerule = random.sample(
            list(range(self.max_fleetsize)), self.fleetsize - 1
        )

    def reset(self) -> "BaseAlpyneEnv.PyObservationType":
        self.shuffle()
        self.stepcounter = 0
        self.config.seed = random.randint(0, 1000)
        self.sim.reset(self.config)
        self.sim.wait_for_completion()
        alpyne_obs = self._catch_dispatcher(
            self._catch_context(self.sim.get_observation())
        )
        obs = self._convert_from_observation(alpyne_obs)
        return obs

    def step(
        self, action: "BaseAlpyneEnv.PyActionType"
    ) -> Tuple["BaseAlpyneEnv.PyObservationType", float, bool, Optional[dict]]:
        self.stepcounter += 1

        if self.stepcounter % 100 == 0:
            self.shuffle()

        alpyne_action = self._convert_to_action(action)
        self.sim.take_action(alpyne_action)

        self.sim.wait_for_completion()

        alpyne_obs = self._catch_dispatcher(self.sim.get_observation())
        obs = self._convert_from_observation(alpyne_obs)
        reward = self._calc_reward(alpyne_obs)
        done = self.sim.is_terminal() or self._terminal_alternative(alpyne_obs)
        info = (
            dict(targetsReached=alpyne_obs.targetsReached)
            if "targetsReached" in alpyne_obs.names()
            else self.sim.last_state[1]
        )  # dictionary of info

        return obs, reward, done, info

    def _catch_context(self, alpyne_obs: Observation) -> Observation:
        if (
            "networkcontext" in alpyne_obs.names()
            and alpyne_obs.networkcontext is not None
            and len(alpyne_obs.networkcontext) > 0
        ):
            self.context = alpyne_obs.networkcontext
        return alpyne_obs

    def _catch_dispatcher(self, alpyne_obs: Observation) -> Observation:
        while alpyne_obs.caller == "Dispatching":
            action = self.dispatcher(alpyne_obs)
            self.sim.take_action(action)
            self.sim.wait_for_completion()
            alpyne_obs = self.sim.get_observation()
        return alpyne_obs

    def _get_observation_space(self) -> spaces.Box:
        """Describe the dimensions and bounds of the observation"""

        obs_sample = self._catch_dispatcher(
            self._catch_context(self.sim.get_observation())
        ).obs
        nStations = len(obs_sample) - self.fleetsize
        shape = ((self.max_fleetsize + nStations) * (len(obs_sample[0]) + 1),)

        return spaces.Box(low=0, high=1, shape=shape)

    def _convert_from_observation(
        self, observation: Observation
    ) -> "BaseAlpyneEnv.PyObservationType":
        """Convert your Observation object to the format expected by Gym"""
        obs = torch.Tensor(observation.obs)
        self.receiver = int(observation.caller)
        n_obs = len(obs) - self.fleetsize + self.max_fleetsize
        obs_len = len(obs[0])
        obs_out = torch.zeros((n_obs, obs_len + 1))
        obs_out[0, 0] = 1
        obs_out[0, 1:] = obs[self.receiver]
        obs = obs[[x != self.receiver for x in range(len(obs))]]
        for i in range(self.max_fleetsize - 1):
            obs_out[i + 1, 1:] = (
                obs[self.shufflerule.index(i)]
                if i in self.shufflerule
                else torch.zeros((obs_len,))
            )
        obs_out[self.max_fleetsize :, 1:] = obs[self.fleetsize :]
        obs_out = obs_out.index_select(
            1, torch.LongTensor([1, 0] + list(range(2, obs_len + 1)))
        )  # relevant for masking
        return obs_out.flatten()

    def _get_action_space(self) -> spaces.Space:
        return spaces.Discrete(5)

    def _convert_to_action(self, action: "BaseAlpyneEnv.PyActionType") -> Action:
        """Convert the action sent as part of the Gym interface to an Alpyne Action object"""
        return Action(
            data=[
                {
                    "name": "actions",
                    "type": "INTEGER_ARRAY",
                    "value": [
                        action,
                    ],
                    "unit": None,
                },
                {
                    "name": "receiver",
                    "type": "INTEGER",
                    "value": self.receiver,
                    "unit": None,
                },
            ]
        )

    def _calc_reward(self, observation: Observation) -> float:
        """Evaluate the performance of the last action based on the current observation"""
        return observation.rew[0]  # if self.stepcounter != self.max_steps else -10

    def _terminal_alternative(self, observation: Observation) -> bool:
        """Optional method to add *extra* terminating conditions"""
        terminal_max_steps = (
            (self.stepcounter >= self.max_steps)
            if self.max_steps is not None
            else False
        )
        time = 0 if self.max_seconds is None else self.run.get_state()[1]["model_time"]
        terminal_max_seconds = (
            time >= self.max_seconds if self.max_seconds is not None else False
        )
        return terminal_max_steps or terminal_max_seconds
