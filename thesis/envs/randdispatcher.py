from alpyne.data.spaces import Action, Observation
import random
import numpy as np


class RandDispatcher:
    def __init__(self, distance=2) -> None:
        self.context = None
        self.distance = distance

    def get_context(self, context):
        nodecontext = [c for c in context if c[-1] == 0]
        self.nodes = dict()
        [self.nodes.update({i: tuple(n[:2])}) for i, n in enumerate(nodecontext)]
        self.nodeCoords = np.array(list(self.nodes.values()))

        pathcontext = [c for c in context if c[-1] != 0]
        self.context = dict()
        [
            self.context.update({self.getClosest(tuple(path[:2])): []})
            for path in pathcontext
        ]
        [
            self.context.update({self.getClosest(tuple(path[2:])): []})
            for path in pathcontext
        ]
        [
            self.context[self.getClosest(tuple(path[:2]))].append(
                self.getClosest(tuple(path[2:]))
            )
            for path in pathcontext
        ]
        [
            self.context[self.getClosest(tuple(path[2:]))].append(
                self.getClosest(tuple(path[:2]))
            )
            for path in pathcontext
        ]

    def getNode(self, last, next):
        next = (
            self.getClosest(tuple(next))
            if next[0] != 0
            else self.getClosest(tuple(last))
        )
        last = self.getClosest(tuple(last))

        for i in range(self.distance):
            possible = list(self.context[next])
            if last in possible:
                possible.pop(possible.index(last))
            last = int(next)
            next = random.choice(possible)
        return next

    def getClosest(self, node):
        index = np.abs(self.nodeCoords - node).mean(1).argmin()
        return index

    def __call__(self, obs: Observation) -> Action:
        if (
            "networkcontext" in obs.names()
            and obs.networkcontext is not None
            and len(obs.networkcontext) > 0
        ):
            self.get_context(obs.networkcontext)
        if self.context is None:
            action_max = obs.n_nodes - 1
            actions = [random.randint(0, action_max) for _ in obs.obs]
        else:
            lasts = [i[1:3] for i in obs.obs]
            nexts = [i[3:5] for i in obs.obs]
            actions = [self.getNode(l, n) for l, n in zip(lasts, nexts)]
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
                    "value": 1,
                    "unit": None,
                },
            ]
        )
