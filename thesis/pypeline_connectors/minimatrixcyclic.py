from stable_baselines3 import PPO
import json
from ..envs.matrix_routing_zoo_cyclic import getMatrixRoutingMACyclic
from ..utils.build_config import build_config
from alpyne.data.spaces import Observation, Action

model_path = "../../models/MiniMatrix_Routing_Attn/PPO-1-1-06_08-22_20_05-42-150000.zip"
hparams_path = "../../models/MiniMatrix_Routing_Attn/PPO-1-1-06_08-22_20_05-42.json"


model = PPO.load(model_path)
with open(hparams_path) as json_file:
    hparams = json.load(json_file)
config = build_config(hparams["env_args"], hparams["fleetsize"])
env = getMatrixRoutingMACyclic(
    fleetsize=hparams["fleetsize"],
    max_fleetsize=hparams["max_fleetsize"],
    config_args=hparams["env_args"],
)


def get_config(id: str):
    if id in config.names():
        return config.values()[config.names().index(id)]
    return "NOT FOUND"


def get_action(observation, caller, n_nodes, context):
    alpyneobs = Observation(
        obs=observation, caller=caller, n_nodes=n_nodes, networkcontext=context
    )
    if caller == "Dispatching":
        return env.dispatcher(alpyneobs)
    action, _ = model.predict(env._convert_from_observation(alpyneobs))
    return env._modify_action(action)


def manual_routing(next, target):
    dx = abs(next[0] - target[0])
    dy = abs(next[1] - target[1])
    if dx > dy:
        if next[0] > target[0]:
            return 1
        if next[0] < target[0]:
            return 3
    if dx < dy:
        if next[1] > target[1]:
            return 4
        if next[1] < target[1]:
            return 2
    return 0


def get_action2(observation, caller, n_nodes, context):
    alpyneobs = Observation(
        obs=observation, caller=caller, n_nodes=n_nodes, networkcontext=context
    )
    if caller == "Dispatching":
        return env.dispatcher(alpyneobs)
    nexts = [tuple(i[3:5]) for i in alpyneobs.obs]
    targets = [tuple(i[5:7]) for i in alpyneobs.obs]
    actions = [manual_routing(n, t) for n, t in zip(nexts, targets)]
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
                "value": 0,
                "unit": None,
            },
        ]
    )
