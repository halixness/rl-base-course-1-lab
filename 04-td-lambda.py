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
#   TD Lambda combines the MonteCarlo (averages reward of encountered states only) method with TD(0).
#   It overcomes the issue of keeping in mind n-steps, therefore you can update the model
#   after just one single step.
#   Plus it is matematically more elegant. There are two new elements:
#   - Elibilities: coefficient to discount gradient descend basically
#   - Lambda: the eligibility coeffiecient to understand how much old gradient to keep      
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

class BaseModel:
  def __init__(self, D):
    self.w = np.random.randn(D) / np.sqrt(D)

  def partial_fit(self, input_, target, eligibility, lr = 1e-2):
    # alpha * difference * eligibility
    # eligibility filters is like a discount factor for new gradient variance
    self.w += lr * (target - input_.dot(self.w)) * eligibility

  def predict(self, X):
    return X.dot(self.w)

# Inspired by https://github.com/dennybritz/reinforcement-learning
class FeatureTransformer:
  def __init__(self, env, n_components = 500, observation_batches = 10000):
    
    # unlimited set of states
    #observation_examples = np.random.random((20000, 4)) * 2 - 2

    # finite set of states
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
  def __init__(self, env, feature_transformer):
    self.env = env
    self.models = []
    self.feature_transformer = feature_transformer

    D = feature_transformer.dimensions

    # the coeffiecients to keep track of gradients in the past
    # for each action there is a set of features
    self.eligibilities = np.zeros((env.action_space.n, D))

    # Each action a model
    for i in range(env.action_space.n):
      model = BaseModel(D)
      self.models.append(model)

  # Transform to linear features with RBF networks
  # predictions: array of G value for each action
  def predict(self, s):
    X = self.feature_transformer.transform([s])
    assert(len(X.shape) == 2) # 2 dimensions
    return np.array([m.predict(X)[0] for m in self.models])

  # partial fit the specific action model
  def update(self, s, a, G, gamma, lambda_):
    X = self.feature_transformer.transform([s])
    assert(len(X.shape) == 2) # 2 dimensions

    self.eligibilities *= gamma * lambda_ # lambda is the eligibility coefficient
    self.eligibilities[a] += X[0] # eligibilities = factors * state input
    self.models[a].partial_fit(X[0], G, self.eligibilities[a])

  # epsilon greedy
  def sample_action(self, s, eps):
    if np.random.random() < eps:
      return self.env.action_space.sample()
    else:
      return np.argmax(self.predict(s))


# returns a list of states_and_rewards, and the total reward
def play_one(model, eps, gamma, lambda_, n=5, display = False):
    
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0

    # Iter steps
    while not done and iters < 10000:

        action = model.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, done, info = env.step(action)

        G = reward + gamma * np.max(model.predict(observation)[0])

        # Improve basing on past rewards
        model.update(prev_observation, action, G, gamma, lambda_)

        totalreward += reward
        iters += 1

        if display:
            env.render()

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
    model = Model(env, ft)
    gamma = 0.9999
    lambda_ = 0.7

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
        totalreward = play_one(model, eps, gamma, lambda_)
        totalrewards[n] = totalreward
        print("episode:", n, "total reward:", totalreward)

    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", -totalrewards.sum())

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(totalrewards)

    play_one(model, eps, gamma, lambda_, display = True)

