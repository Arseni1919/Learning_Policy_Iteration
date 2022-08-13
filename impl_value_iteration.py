from impl_gamblers_problem import GamblersProblemEnv
from functions import *


def update_axes(ax, input_dict, title=''):
    ax.cla()
    x_p = list(input_dict.keys())
    y_p = list(input_dict.values())
    ax.plot(x_p, y_p)
    ax.set_title(f'{title}')


def plot_results(policy_dict, v_func_dict, title='', to_plot=True):
    # print()
    # print(input_dict)
    if to_plot:
        update_axes(ax1, policy_dict, title='policy')
        update_axes(ax2, v_func_dict, title='v_func')

        plt.pause(0.01)
        # plt.show()


def value_iteration(env, policy, v_func):
    epsilon = 1e-10
    delta = epsilon + 1

    iteration = 1
    while delta > epsilon:
        iteration += 1
        delta = 0

        states = env.get_states()
        for state in states:
            v = v_func[state]

            # get action with max value
            new_v_dict = {}
            actions = env.get_actions(state)
            for action in actions:
                new_v_dict[action] = 0
                states_plus = env.get_states_plus()
                for next_state in states_plus:
                    prob, reward = env.get_prob(state, action, next_state)
                    new_v_dict[action] += prob * (reward + GAMMA * v_func[next_state])

            max_action = max(new_v_dict, key=new_v_dict.get)
            max_value = new_v_dict[max_action]

            policy[state] = max_action
            v_func[state] = max_value
            delta = max(delta, np.abs(v - max_value))

            print(f'\r[iter {iteration}] delta: {delta}', end='')

        # plot_results(policy, title='policy', to_plot=False)
        # plot_results(v_func, title='v_func', to_plot=False)
        plot_results(policy, v_func, to_plot=True)

    return policy, v_func


def main():
    # env
    env = GamblersProblemEnv(p_h=P_h)

    # init
    policy = env.init_policy()
    v_func = env.init_v_func()

    # run
    policy, v_func = value_iteration(env, policy, v_func)

    # plots
    plot_results(policy, v_func)
    plt.show()


if __name__ == '__main__':
    GAMMA = 1
    # P_h = 0.4
    # P_h = 0.25
    P_h = 0.55
    fig, (ax1, ax2) = plt.subplots(1, 2)
    main()


