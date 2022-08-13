import random

import numpy as np


class GamblersProblemEnv:
    def __init__(self, p_h):
        self.p_h = p_h
        self.goal = 100
        self.min_capital = 1
        self.states = np.arange(self.min_capital, self.goal)  # capital

    def get_prob(self, state, action, other_state):
        """
        reward if fully defined by state, action and next state
        :returns: prob, reward
        """
        # if state == 50 and other_state == 100 and action == 50:
        #     print('state')
        # HEADS
        new_state = state + action
        if new_state > self.goal:
            raise RuntimeError('new_state > self.goal')
        if new_state == other_state:
            if new_state == self.goal:
                return self.p_h, 1
            else:
                return self.p_h, 0

        # TAILS
        new_state = state - action
        if new_state == other_state:
            return 1 - self.p_h, 0
        return 0.0, 0

    def get_states(self):
        return self.states

    def get_states_plus(self):
        return np.arange(self.min_capital - 1, self.goal + 1)

    def get_actions(self, state):
        max_stake = min(state, 100 - state)
        return np.arange(max_stake + 1)

    def init_policy(self):
        return {state: 0 for state in self.states}

    def init_v_func(self):
        return {state: 0 for state in self.get_states_plus()}

