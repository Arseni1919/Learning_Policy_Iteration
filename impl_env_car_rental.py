import numpy as np
from scipy.stats import poisson
import itertools


class JacksCarRentalEnv:
    def __init__(self, max_cars=10):
        self.rent_per_car = 10
        self._cost_per_move = 2
        self.max_cars = max_cars
        self.max_cars_move_per_night = 5
        self.L_REQ_1 = 3
        self.L_REQ_2 = 4
        self.L_RET_1 = 3
        self.L_RET_2 = 2

    def reset(self):
        pass

    def plot(self, text):
        print(text)

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
        curr_prob = 0.0

        # take an action
        if 0 < action <= 5:
            to_move = min(curr_1, action)
            to_move = min(to_move, self.max_cars - curr_2)
            curr_1 -= to_move
            curr_2 += to_move
        if -5 <= action < 0:
            action *= -1
            to_move = min(curr_2, action)
            to_move = min(to_move, self.max_cars - curr_1)
            curr_2 -= to_move
            curr_1 += to_move
        curr_reward += - to_move * self._cost_per_move
        self.plot(f'to move: {to_move}, reward: {curr_reward}')

        other_1, other_2 = other_state

        # for 1
        min_num_1 = min(curr_1, other_1)
        min_num_1 = max(min_num_1, 1)
        for i in range(min_num_1):

            rental_cars_1 = curr_1 - i
            prob_req_1 = poisson.pmf(k=rental_cars_1, mu=self.L_REQ_1)

            return_cars_1 = other_1 - i
            prob_ret_1 = poisson.pmf(k=return_cars_1, mu=self.L_RET_1)

            # for 2
            min_num_2 = min(curr_2, other_2, 1)
            min_num_2 = max(min_num_2, 1)
            for j in range(min_num_2):
                rental_cars_2 = curr_2 - j
                prob_req_2 = poisson.pmf(k=rental_cars_2, mu=self.L_REQ_2)

                return_cars_2 = other_2 - j
                prob_ret_2 = poisson.pmf(k=return_cars_2, mu=self.L_RET_2)

                chance = prob_req_1 * prob_req_2 * prob_ret_1 * prob_ret_2

                curr_prob += chance
                curr_reward += self.rent_per_car * (rental_cars_1 + rental_cars_2) * chance

        return curr_prob, curr_reward


# a = np.random.poisson(self.L_REQ_1, 1)[0]
# a = np.random.poisson(self.L_REQ_2, 1)[0]
# a = np.random.poisson(self.L_RET_1, 1)[0]
# a = np.random.poisson(self.L_RET_2, 1)[0]
