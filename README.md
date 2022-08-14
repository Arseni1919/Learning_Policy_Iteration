# Learning Policy Iteration Algorithm and Value Iteration Algorithm

## Under The Umbrella Of Dynamic Programming

### Policy Iteration

```python
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

def policy_iteration(env, policy, v_func):
    policy_stable, v_func_stable = False, False

    iteration = 0
    while not policy_stable or not v_func_stable:
        iteration += 1
        print(f'\n###\niteration {iteration}:\n###\n')
        v_func = policy_evaluation(policy, v_func, env)
        policy_stable = policy_improvement(policy, v_func, env)

        plot_results(policy, title='policy', to_plot=False)
        plot_results(v_func, title='v_func', to_plot=False)
```

### Value Iteration

```python
def value_iteration(env, policy, v_func):
    epsilon = 1e-20
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
            
        plot_results(policy, v_func, to_plot=True)

    return policy, v_func
```

## Credits

- Sutton and Barto - RL: An Introduction