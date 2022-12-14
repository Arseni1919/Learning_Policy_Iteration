import itertools
from impl_env_car_rental import JacksCarRentalEnv
from functions import *


def plot_results(mat, title='', to_plot=True):
    print()
    print(mat)
    if to_plot:
        plt.imshow(mat)
        plt.title(f'{title}')
        plt.show()


def policy_evaluation(policy, v_func_old, env):
    print('\n policy_evaluation')
    epsilon = 0.01
    delta = epsilon + 1
    v_func = np.copy(v_func_old)

    iterations = 0
    while delta > epsilon:
        iterations += 1
        delta = 0
        # for each state
        for state in itertools.product(range(env.max_cars), repeat=2):
            curr_v = v_func[state]
            action = policy[state]
            # calc new v
            new_v = 0
            for other_state in itertools.product(range(env.max_cars), repeat=2):
                curr_prob, curr_reward = env.get_prob(state, action, other_state)
                new_v += curr_prob * (curr_reward + GAMMA * v_func[other_state])
            v_func[state] = new_v

            delta = max(delta, np.abs(curr_v - v_func[state]))
            print(f'\r[iter {iterations}] state: {state}, delta: {delta}', end='')

    return v_func


def policy_improvement(policy, v_func, env):
    print('\n policy_improvement')
    policy_stable = True
    for state in itertools.product(range(env.max_cars), repeat=2):

        old_action = policy[state]

        action_dict = {}
        for action in range(-5, 6):
            v_sum = 0
            for other_state in itertools.product(range(env.max_cars), repeat=2):
                curr_prob, curr_reward = env.get_prob(state, action, other_state)
                v_sum += curr_prob * (curr_reward + GAMMA * v_func[other_state])
            action_dict[action] = v_sum

        best_action = max(action_dict.keys(), key=action_dict.get)
        print(f'\rstate: {state}, action: {best_action}, policy_stable: {policy_stable}', end='')
        policy[state] = best_action
        if old_action != policy[state]:
            policy_stable = False
    return policy_stable


def termination_check():
    value_func_stable = True
    return value_func_stable


def policy_iteration(env, policy, v_func):
    policy_stable, v_func_stable = False, False

    iteration = 0
    while not policy_stable or not v_func_stable:
        iteration += 1
        print(f'\n###\niteration {iteration}:\n###\n')
        v_func = policy_evaluation(policy, v_func, env)
        policy_stable = policy_improvement(policy, v_func, env)
        v_func_stable = termination_check()

        plot_results(policy, title='policy', to_plot=False)
        plot_results(v_func, title='v_func', to_plot=False)

    # plot_results(policy, title='policy')
    # plot_results(v_func, title='v_func')

    return policy, v_func


def main():
    # env
    env = JacksCarRentalEnv(MAX_CARS, VERSION_2)
    max_cars = env.max_cars

    # init
    policy = np.zeros((max_cars, max_cars))
    v_func = np.zeros((max_cars, max_cars))

    # policy iteration
    policy, v_func = policy_iteration(env, policy, v_func)

    plot_results(policy, title='policy')
    plot_results(v_func, title='v_func')


if __name__ == '__main__':
    GAMMA = 0.9
    MAX_CARS = 10
    # VERSION_2 = False
    VERSION_2 = True
    main()

