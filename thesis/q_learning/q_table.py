import numpy as np

class QTable:
    def __init__(self, n_actions):
        self._n_actions = n_actions
        self.states = []
        self.table = np.zeros((0,n_actions))
    def __getitem__(self, sliced):
        assert len(sliced)==2
        assert not isinstance(sliced[0], slice)
        if sliced[0] not in self.states:
            self._addState(sliced[0])
        return self.table[self.states.index(sliced[0])][sliced[1]]
    def __setitem__(self, sliced, value):
        assert len(sliced)==2
        assert not isinstance(sliced[0], slice)
        if sliced[0] not in self.states:
            self._addState(sliced[0])
        self.table[self.states.index(sliced[0])][sliced[1]] == value
    def __repr__(self):
        return str(self.table)
    def __len__(self):
        return len(self.states)
    def _addState(self, state):
        self.states.append(state)
        self.table = np.append(self.table, np.zeros((1,self._n_actions)), axis = 0)