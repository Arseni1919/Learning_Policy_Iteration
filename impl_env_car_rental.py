import numpy as np
from scipy.stats import poisson, skellam
import itertools


class JacksCarRentalEnv:
    def __init__(self, max_cars=10, version_2=False):
        self.rent_per_car = 10
        self._cost_per_move = 2
        self.max_cars = max_cars
        self.version_2 = version_2
        self.max_cars_move_per_night = 5
        self.L_REQ_1 = 3
        self.L_REQ_2 = 4
        self.L_RET_1 = 3
        self.L_RET_2 = 2

    def reset(self):
        pass

    def plot(self, text):
        pass
        # print(text)

    def step(self, action):
        """
        time step - a day
        state: [cars at 1, cars at 2]

        action:
        action = 0 -> do nothing
        0 < action <= 5 -> move cars from 1 to 2
        -5 <= action < 0 -> move cars from 2 to 1
        """
        state, reward = [0, 0], 0
        return state, reward

    def get_prob(self, state, action, other_state):
        """
        reward if fully defined by state, action and next state
        """
        curr_1, curr_2 = state
        to_move = 0
        curr_reward = 0
        # curr_prob = 0.0

        # take an action
        # from 1 to 2
        if 0 < action <= 5:
            to_move = min(curr_1, action)
            to_move = min(to_move, self.max_cars - curr_2)
            curr_1 -= to_move
            curr_2 += to_move

            if self.version_2:
                if to_move > 0:
                    to_move -= 1

        # from 2 to 1
        if -5 <= action < 0:
            action *= -1
            to_move = min(curr_2, action)
            to_move = min(to_move, self.max_cars - curr_1)
            curr_2 -= to_move
            curr_1 += to_move

        curr_reward += - to_move * self._cost_per_move
        if self.version_2:
            if curr_1 > 10:
                curr_reward += 4
            if curr_2 > 10:
                curr_reward += 4

        other_1, other_2 = other_state

        # for 1
        diff_1 = other_1 - curr_1
        prob_1 = skellam.pmf(k=diff_1, mu1=self.L_RET_1, mu2=self.L_REQ_1)
        if diff_1 < 0:
            curr_reward += 10 * (-1) * diff_1

        # for 2
        diff_2 = other_2 - curr_2
        prob_2 = skellam.pmf(k=diff_2, mu1=self.L_RET_2, mu2=self.L_REQ_2)
        if diff_2 < 0:
            curr_reward += 10 * (-1) * diff_2

        curr_prob = prob_1 * prob_2

        return curr_prob, curr_reward


# a = np.random.poisson(self.L_REQ_1, 1)[0]
# a = np.random.poisson(self.L_REQ_2, 1)[0]
# a = np.random.poisson(self.L_RET_1, 1)[0]
# a = np.random.poisson(self.L_RET_2, 1)[0]
