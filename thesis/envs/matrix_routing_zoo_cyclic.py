from alpyne.client.alpyne_client import AlpyneClient
from .matrix_routing_centralized import MatrixRoutingCentral
from pettingzoo import ParallelEnv
from gym import spaces
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.util import (
    copy_obs_dict,
    dict_to_obs,
    obs_space_info,
)
from collections import OrderedDict
from copy import deepcopy
from stable_baselines3.common.monitor import Monitor

counter = [0]
satellite_counter = [0]


def getMatrixRoutingMACyclic(verbose = False, **kwargs):
    kwargs["verbose"] = verbose
    raw = MatrixRoutingMACyclic(**kwargs)
    wrapped = ZooToVec_POST(Monitor(ZooToVec_PRE(raw)), verbose = verbose)
    return wrapped


class ZooToVec_POST(DummyVecEnv):
    def __init__(self, wrapped_env: Monitor, verbose):
        self.envs = [
            wrapped_env,
        ]
        env = wrapped_env.env.envs[0]
        self.num_envs = self.num_agents = env.num_agents
        self.actions = None
        self.verbose = verbose
        obs_space = env.observation_space(env.agents[0])
        VecEnv.__init__(
            self,
            self.num_envs,
            obs_space,
            env.action_space(env.agents[0]),
        )
        self.keys, shapes, dtypes = obs_space_info(obs_space)
        self.buf_obs = OrderedDict(
            [
                (k, np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]))
                for k in self.keys
            ]
        )
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]

    def reset(self):
        obs = self.envs[0].reset()
        for agent in range(self.num_envs):
            self._save_obs(agent, obs)
        return self._obs_from_buf()

    def step_wait(self):
        # print(self.actions)
        # print(self.action_masks())
        if self.verbose:
            print(f"ZTV_actions{self.actions}")
        step_ret, _, all_done, ep_info = self.envs[0].step(self.actions)
        obs = np.array(step_ret["obs"])
        rew = np.array(step_ret["rew"])
        dones = np.array(step_ret["done"], dtype=bool)
        infos = step_ret["info"]
        for info in infos:
            for key, val in ep_info.items():
                info[key] = val
        if all_done:
            obs = self.reset()

        for agent in range(self.num_envs):
            self._save_obs(agent, obs)
        # print(obs)

        if self.verbose:
            print(f"ZTV_obs{obs}")
            print(f"ZTV_obs{rew}")
            print(f"ZTV_obs{dones}")
        return obs, rew, dones, infos

    def _save_obs(self, env_idx, obs):
        self.buf_obs[None][env_idx] = obs[env_idx]
        if self.verbose:
            print(f"ZTV_buf_obs{self.buf_obs}")

    def action_masks(self):
        mask = np.ones((self.buf_obs[None].shape[0], 5))
        for i in range(4):
            mask[:, i + 1] = np.all(
                self.buf_obs[None][:, 2 * i + 9 : 2 * i + 11] == 0, axis=1
            )
        return mask

    def _get_target_envs(self, indices):
        indices = self._get_indices(indices)
        return [self for i in indices]

    def get_attr(self, attr_name: str, indices=None):
        if attr_name != "action_masks":
            return super().get_attr(attr_name, indices)
        else:
            if indices is None:
                indices = range(self.num_envs)
            return [lambda: self.action_masks()[i] for i in indices]

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        if method_name != "action_masks":
            target_envs = self._get_target_envs(indices)
            return [
                getattr(env_i, method_name)(*method_args, **method_kwargs)
                for env_i in target_envs
            ]
        else:
            indices = self._get_indices(indices)
            return self.action_masks()[indices]


class ZooToVec_PRE(DummyVecEnv):
    """Wrapper for SB3, but does not work yet (At least not if additionally wrapped in monitor)"""

    def __init__(self, parallel_env: ParallelEnv):
        self.num_envs = self.num_agents = parallel_env.num_agents
        self.envs = [
            parallel_env,
        ]
        obs_space = parallel_env.observation_space(parallel_env.agents[0])
        VecEnv.__init__(
            self,
            self.num_envs,
            obs_space,
            parallel_env.action_space(parallel_env.agents[0]),
        )
        self.keys, shapes, dtypes = obs_space_info(obs_space)
        self.buf_obs = OrderedDict(
            [
                (k, np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]))
                for k in self.keys
            ]
        )
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.metadata = parallel_env.metadata

    def reset(self):
        obs = self.envs[0].reset()
        for agent in range(self.num_envs):
            self._save_obs(agent, obs)
        return self._obs_from_buf()

    def step_async(self, actions: np.ndarray) -> None:
        actions = {str(key): value for key, value in enumerate(actions)}
        return super().step_async(actions)

    def step_wait(self):
        obs, rews, dones, infos = self.envs[0].step(self.actions)
        self.buf_rews = list(rews.values())
        self.buf_dones = list(dones.values())
        self.buf_infos = deepcopy(list(infos.values()))

        if np.all(self.buf_dones):
            for agent in range(self.num_envs):
                self.buf_infos[agent]["terminal_observation"] = obs[str(agent)]

        for agent in range(self.num_envs):
            self._save_obs(agent, obs)

        return (
            dict(
                obs=self._obs_from_buf(),
                rew=self.buf_rews,
                done=self.buf_dones,
                info=deepcopy(self.buf_infos),
            ),
            np.mean(self.buf_rews),
            np.all(self.buf_dones),
            dict(),
        )

    # def _obs_from_buf(self):
    #     return self.buf_obs

    def _save_obs(self, env_idx, obs):
        self.buf_obs[None][env_idx] = obs[str(env_idx)]


class MatrixRoutingMACyclic(ParallelEnv):
    """Pettingzoo Parallelenv Wrapper for the MatrixRoutingCentral, corresponds to Markov Game"""

    def __init__(
        self,
        model_path: str = None,
        port=51150,
        client: AlpyneClient = None,
        max_steps: int = None,
        max_seconds: int = None,
        fleetsize: int = 1,
        max_fleetsize: int = None,
        config_args: dict = dict(),
        dispatcher=None,
        counter=None,
        verbose: bool = False,
    ):
        config_args.update(reward_separateAgv=True)
        self.gym_env = MatrixRoutingCentral(
            model_path,
            port,
            client,
            max_steps,
            max_seconds,
            fleetsize,
            max_fleetsize,
            config_args,
            dispatcher,
            counter,
            do_shuffle=False,
            verbose=verbose,
        )

        self.metadata = dict(is_parallelizable=True)

        # self.action_buffer = []
        self.max_fleetsize = max_fleetsize

        self.possible_agents = self.agents = [str(i) for i in range(fleetsize)]
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: dict() for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        # self.agent_selection = None
        self.observation_spaces = {
            agent: self._get_observation_space(agent) for agent in self.agents
        }
        self.action_spaces = {
            agent: self._get_action_space(agent) for agent in self.agents
        }

        self.reset()

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent) -> spaces.Space:
        return self.action_spaces[agent]

    def _get_observation_space(self, agent):
        return spaces.Box(
            0, 1, (self.gym_env.observation_space.shape[0] + self.max_fleetsize,)
        )  # Not nice, I know...

    def _get_action_space(self, agent):
        return spaces.Discrete(5)

    def step(self, action_in):
        # self.action_buffer.append(action)
        # if len(self.action_buffer) == len(self.agents):
        #     actions = np.zeros((self.max_fleetsize))
        #     actions[self.gym_env.shufflerule] = self.action_buffer
        #     obs, reward, done, info = self.gym_env.step(actions)
        #     self.action_buffer.clear()
        #     self.save_observation(obs, reward, done)
        # self.agent_selection = self.agents[len(self.action_buffer)]
        actions = np.zeros((self.max_fleetsize,))
        actions[self.gym_env.shufflerule] = list(action_in.values())
        obs, reward, done, info = self.gym_env.step(actions)
        self.save_observation(obs, reward, done)
        return self.observations, self.rewards, self.dones, self.infos

    def reset(self, **kwargs):
        obs = self.gym_env.reset()
        self.save_observation(obs)
        # self.action_buffer.clear()
        # self.agent_selection = self.agents[0]
        # return self.observe(self.agent_selection)
        return self.observations

    # def last(self, observe=True):
    #     obs = self.observe(self.agent_selection) if observe else None
    #     reward = self.rewards[self.agent_selection]
    #     done = self.dones[self.agent_selection]
    #     info = self.infos[self.agent_selection]
    #     return obs, reward, done, info

    def save_observation(self, obs, rewards=0, done=False):
        obs = obs.reshape(self.max_fleetsize, obs.shape[-1] // self.max_fleetsize)
        for i, agent in zip(self.gym_env.shufflerule, self.agents):
            self.observations[agent] = self.transform_obs(obs, i)
            self.dones[agent] = done
            if isinstance(rewards, int):
                self.rewards[agent] = rewards
            else:
                self.rewards[agent] = rewards[int(agent)]

    def transform_obs(self, obs, i):
        inSystem = obs[i, 0] == 1
        await_orders = obs[i, 1] == 1
        transformed = np.zeros((obs.shape[0], obs.shape[1] + 1), dtype=np.float32)
        if inSystem:
            transformed[0, 0] = 1
            transformed[0, 1:] = obs[i]
            transformed[1:, 1:] = obs[[j for j in range(obs.shape[0]) if j != i]]
        return transformed.flatten()

    # def observe(self, agent: str):
    #     return self.observations[agent]
