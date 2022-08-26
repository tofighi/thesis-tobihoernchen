from thesis.envs.matrix_routing_zoo import MatrixRoutingMA
from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env


def setup_matrix_for_ray(verbose=False):
    env_fn = lambda config: MatrixRoutingMA(
        model_path="../../envs/MiniMatrix.zip", verbose=verbose, **config
    )
    register_env("matrix", lambda config: PettingZooEnv(env_fn(config)))
