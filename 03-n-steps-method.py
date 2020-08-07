#
# Code provided by :
#
#   https://deeplearningcourses.com/c/deep-reinforcement-learning-in-python
#   https://www.udemy.com/deep-reinforcement-learning-in-python
#
# Rewritten and commented by:
#   
#   Diego Calanzone - University of Parma
#   August 2020
#
#   N-steps method is a TD algorithm where multiple steps are fed into the network.
#   This results in better G estimation, however it does not avoid the sequence order bias.
#   
#

from __future__ import print_function, division
from builtins import range

import gym
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gym import wrappers
from datetime import datetime
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor

class SGDRegressor:
  def __init__(self):
    self.w = None
    self.lr = 1e-2

  def partial_fit(self, X, Y):
    if self.w is None:
      D = X.shape[1]
      self.w = np.random.randn(D) / np.sqrt(D)
    self.w += self.lr*(Y - X.dot(self.w)).dot(X)

  def predict(self, X):
    if self.w is None:
      D = X.shape[1]
      self.w = np.random.randn(D) / np.sqrt(D)
    return X.dot(self.w)

# Inspired by https://github.com/dennybritz/reinforcement-learning
class FeatureTransformer:
  def __init__(self, env, n_components = 500, observation_batches = 10000):
    observation_examples = np.array([env.observation_space.sample() for x in range(observation_batches)])
    scaler = StandardScaler()
    scaler.fit(observation_examples)

    # Used to converte a state to a featurizes represenation.
    # We use RBF kernels with different variances to cover different parts of the space
    featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=n_components)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=n_components)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=n_components)),
            ("rbf4", RBFSampler(gamma=2.0, n_components=n_components))
            ])
    example_features = featurizer.fit_transform(scaler.transform(observation_examples))

    self.dimensions = example_features.shape[1]
    self.scaler = scaler
    self.featurizer = featurizer

  def transform(self, observations):
    # print "observations:", observations
    scaled = self.scaler.transform(observations)
    # assert(len(scaled.shape) == 2)
    return self.featurizer.transform(scaled)


# Holds one SGDRegressor for each action
class Model:
  def __init__(self, env, feature_transformer, learning_rate):
    self.env = env
    self.models = []
    self.feature_transformer = feature_transformer

    # Each action a model
    for i in range(env.action_space.n):
      model = SGDRegressor()
      self.models.append(model)

  # Transform to linear features with RBF networks
  # predictions: array of G value for each action
  def predict(self, s):
    X = self.feature_transformer.transform(np.atleast_2d(s))
    return np.array([m.predict(X)[0] for m in self.models])

  # partial fit the specific action model
  def update(self, s, a, G):
    X = self.feature_transformer.transform(np.atleast_2d(s))
    self.models[a].partial_fit(X, [G])

  # epsilon greedy
  def sample_action(self, s, eps):
    if np.random.random() < eps:
      return self.env.action_space.sample()
    else:
      return np.argmax(self.predict(s))


# returns a list of states_and_rewards, and the total reward
def play_one(model, eps, gamma, n=5, display = False):
    
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0

    rewards = []
    states = []
    actions = []

    # array of gamma^0 * gamma^1 * ... * gamma^(n-1)
    multiplier = np.array([gamma] * n)**np.arange(n)

    # Iter steps
    while not done and iters < 10000:

        action = model.sample_action(observation, eps)

        # save states and actions
        states.append(observation)
        actions.append(action)

        prev_observation = observation
        observation, reward, done, info = env.step(action)

        # save reward
        rewards.append(reward)

        # update the model as we get N steps
        if len(rewards) >= n:

            # Compute ewards
            return_up_to_prediction = multiplier.dot(rewards[-n:])

            # G function for all the rewards
            G = return_up_to_prediction + (gamma ** n) * np.max(model.predict(observation)[0])

            # Improve basing on past rewards
            model.update(states[-n], actions[-n], G)

        totalreward += reward
        iters += 1

        if display:
            env.render()

    ###

    # empty cache -> keep last n+1 rewards
    if n == 1:
        rewards = []
        states = []
        actions = []
    else:
        rewards = rewards[-n+1:]
        states = states[-n+1:]
        actions = actions[-n+1:]

    # if we hit the goal
    if observation[0] >= 0.5:

        # learn remaining rewards and empty the cache 
        while len(rewards) > 0:
            # compute G
            G = multiplier[:len(rewards)].dot(rewards)
            # train the net
            model.update(states[0], actions[0], G)
            # remove
            rewards.pop(0)
            states.pop(0)
            actions.pop(0)

    # if we did not hit the goal
    else:
        # learn the remaining rewards as negative (model has failed, bad guy!) and empty the cache
        while len(rewards) > 0:
            guess_rewards = rewards + [-1] * (n - len(rewards))
            G = multiplier.dot(guess_rewards)
            model.update(states[0], actions[0], G)
            rewards.pop(0)
            states.pop(0)
            actions.pop(0)

    return totalreward

def plot_running_avg(totalrewards):
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
  plt.plot(running_avg)
  plt.title("Running Average")
  plt.show()

if __name__ == '__main__':

    # Env setting
    env = gym.make('MountainCar-v0')
    ft = FeatureTransformer(env)
    model = Model(env, ft, "constant")
    gamma = 0.99

    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)

    # Env hyperparams
    N = 300
    totalrewards = np.empty(N)
    costs = np.empty(N)

    for n in range(N):
        # Nice epsilon greedyyy
        eps = 0.1*(0.97**n)
        totalreward = play_one(model, eps, gamma)
        totalrewards[n] = totalreward
        print("episode:", n, "total reward:", totalreward)

    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", -totalrewards.sum())

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(totalrewards)

    play_one(model, eps, gamma, display = True)

