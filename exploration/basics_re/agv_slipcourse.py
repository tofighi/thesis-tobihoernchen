from alpyne.client.alpyne_client import AlpyneClient
import numpy as np
import random
import time
from IPython.display import clear_output

client = AlpyneClient(
    "models\AGV_SlipCourse Exported\AGV_SlipCourse.zip", blocking=True, port=51151
)

config = client.configuration_template
config.inSlipChance = 0.3
run = client.create_reinforcement_learning(config)

run = run.run()
obs_hist = []
while not run.is_terminal():
    obs = run.get_observation()
    obs_hist.append(obs.values())
    print(obs)
    action = client.action_template
    action.inAction = int(input("0-forward, 1-back, 2-wait, 3-toggle: "))
    run.take_action(action)
