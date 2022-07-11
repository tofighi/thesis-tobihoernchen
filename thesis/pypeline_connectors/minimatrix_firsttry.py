from stable_baselines3 import PPO
from ..envs.minimatrix_firsttry import MiniMatrix

model_path = "../../models/MiniMatrix/PPO-1-None-1657039235.6298256-42-200000"
model = PPO.load(model_path)
env = MiniMatrix()


def get_action(observation):
    action, _ = model.predict(env._modify_observation(observation))
    return env._modify_action(action)
