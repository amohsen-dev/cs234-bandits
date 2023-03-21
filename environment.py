import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
nest = tf.nest


class ClinicalEnv(py_environment.PyEnvironment):

    def __init__(self, data, episode_length=1, sensitive_reward=False):
        super().__init__()
        n_features = data.shape[1]-1
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(n_features,), dtype=np.float32, minimum=0, name='observation')
        self._state = np.zeros(n_features)
        self._target = None
        self._trial = 0
        self._episode = -1
        self._episode_length = episode_length
        self._trials_per_episode = episode_length * self._action_spec.num_values
        self._episode_ended = False
        self._data = data
        self._sensitive_reward = sensitive_reward

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._trial = 0
        self._episode += 1
        self._state = self._data.iloc[self._episode * self._trials_per_episode + self._trial, :-1].values.astype(np.float32)
        self._target = self._data.iloc[self._episode * self._trials_per_episode + self._trial, -1]
        self._episode_ended = False
        return ts.restart(self._state)

    def _step(self, action):
        #  print(self._episode, self._trial, self._episode * self._trials_per_episode + self._trial)
        self._trial += 1
        if self._sensitive_reward:
            reward = - (action - self._target) ** 2
        else:
            reward = 0 if action == self._target else -1
        if self._episode_ended:
            return self.reset()
        if self._trial > self._episode_length:
            self._episode_ended = True
        else:
            new_state = self._data.iloc[self._episode * self._trials_per_episode + self._trial, :-1].values.astype(np.float32)
            self._state = new_state
        if self._episode_ended:
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward, discount=1.0)
