import gym
import numpy as np
from gym import wrappers

env = gym.make('CartPole-v0')

state = env.reset()
episodes = 100
weights_adjustments = 100

best_ep_avg = -99

best_weights = None

# Weights adjustements
for x in range(weights_adjustments):

    new_weights = np.random.rand(len(state))
    episode_durations = []

    # Play multiple episodes
    for i in range(episodes):

        steps = 0
        done = False
        state = env.reset()

        # Play episode
        while not done:

            # Compute action by weights
            r = np.dot(state, new_weights)

            if r > 0:
                action = 1
            else:
                action = 0

            new_state, reward, done, info = env.step(action)
            state = new_state

            steps += 1

        # Save length
        episode_durations.append(steps)

    # Episode avg
    ep_avg = np.average(episode_durations)
    print("=> Episode Average: {}".format(ep_avg))
    print("~~> Best avg: {}".format(best_ep_avg))
    print("")

    if ep_avg > best_ep_avg:
        best_ep_avg = ep_avg
        best_weights = new_weights

# Play with best weights 
print("best weights")
print(best_weights)

done = False
state = env.reset()

# Play episode
while not done:

    # Compute action by weights
    r = np.dot(state, best_weights)

    if r > 0:
        action = 1
    else:
        action = 0

    new_state, reward, done, info = env.step(action)
    state = new_state
    env.render()

env.close()