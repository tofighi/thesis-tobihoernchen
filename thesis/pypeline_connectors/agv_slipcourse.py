from stable_baselines3 import PPO

model_path = "../../models/AgvSlipCourse/PPO/PPO-1653220759.1479478-42-100000"
model = PPO.load(model_path)


def get_action(observation):
    action, _ = model.predict(observation)
    return action
