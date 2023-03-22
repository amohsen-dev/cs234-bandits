import abc
import numpy as np
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.bandits.agents import lin_ucb_agent
from tf_agents.bandits.agents import neural_linucb_agent
from tf_agents.bandits.agents import linear_thompson_sampling_agent
from tf_agents.bandits.metrics import tf_metrics
from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.networks import network
nest = tf.nest

from environment import ClinicalEnv


class Learner:
    def __init__(self, raw_data, seed=234, sensitive_reward=False):
        self.data = raw_data.sample(raw_data.shape[0], random_state=seed).reset_index(drop=True).copy()
        clinical_env = ClinicalEnv(self.data, episode_length=1, sensitive_reward=sensitive_reward)
        self.tf_clinical_env = tf_py_environment.TFPyEnvironment(clinical_env)
        self.observation_spec = tensor_spec.TensorSpec([self.data.shape[1] - 1], tf.float32)
        self.time_step_spec = ts.time_step_spec(self.observation_spec)
        self.action_spec = tensor_spec.BoundedTensorSpec(dtype=tf.int32, shape=(), minimum=0, maximum=2)
        self.agent = None

    @abc.abstractmethod
    def create_agent(self, **kwargs):
        pass

    @staticmethod
    def compute_optimal_reward(observation):
        return 0

    def learn(self):
        if self.agent is None:
            raise Exception('Must create agent first. e.g. call create_agent')
        num_iterations = self.data.shape[0]

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.policy.trajectory_spec,
            batch_size=1,
            max_length=1)

        regret_metric = tf_metrics.RegretMetric(Learner.compute_optimal_reward)
        observers = [replay_buffer.add_batch, regret_metric]

        driver = dynamic_step_driver.DynamicStepDriver(
            env=self.tf_clinical_env,
            policy=self.agent.collect_policy,
            num_steps=1,
            observers=observers)

        regret_values = []

        tss, ps = None, None
        for _ in range(num_iterations):
            try:
                tss, ps = driver.run(tss, ps)
                _ = self.agent.train(replay_buffer.gather_all())
                replay_buffer.clear()
                regret_values.append(regret_metric.result())
            except Exception:
                break
        return np.array(regret_values)


class LinUCBLearner(Learner):
    def __int__(self, raw_data, seed=234, sensitive_reward=False):
        super().__init__(raw_data, seed, sensitive_reward)

    def create_agent(self, **kwargs):
        self.agent = lin_ucb_agent.LinearUCBAgent(time_step_spec=self.time_step_spec, action_spec=self.action_spec)


class LinearThompsonSamplingLearner(Learner):
    def __int__(self, raw_data, seed=234, sensitive_reward=False):
        super().__init__(raw_data, seed, sensitive_reward)

    def create_agent(self, **kwargs):
        alpha = kwargs.get('alpha', .2)
        gamma = kwargs.get('gamma', 1.)
        tikhonov_weight = kwargs.get('tikhonov_weight', 1.)
        self.agent = linear_thompson_sampling_agent.LinearThompsonSamplingAgent(
            time_step_spec=self.time_step_spec,
            action_spec=self.action_spec,
            alpha=alpha,
            gamma=gamma,
            tikhonov_weight=tikhonov_weight
        )


class NeuralLinUCBLearner(Learner):
    def __int__(self, raw_data, seed=234, sensitive_reward=False):
        super().__init__(raw_data, seed, sensitive_reward)

    class SimpleNet(network.Network):

        def __init__(self, observation_spec, encoding_dim, hidden_neurons, dropout=.5):
            super().__init__(observation_spec, state_spec=(), name='SimpleNet')
            self._encoding_dim = encoding_dim
            self._hidden_neurons = hidden_neurons
            context_dim = observation_spec.shape[0]
            self._nn_layers = [
                tf.keras.layers.Dense(hidden_neurons, input_shape=(context_dim,), activation='relu'),
                tf.keras.layers.Dropout(rate=dropout),
                tf.keras.layers.Dense(encoding_dim)
            ]

        def call(self, inputs, step_type=None, network_state=()):
            del step_type
            inputs = tf.cast(inputs, tf.float32)
            for layer in self._nn_layers:
                inputs = layer(inputs)
            return inputs, network_state

    def create_agent(self, **kwargs):
        alpha = kwargs.get('alpha', 1)
        gamma = kwargs.get('gamma', 1.)
        encoding_dim = kwargs.get('encoding_dim', 4)
        hidden_neurons = kwargs.get('hidden_neurons', 4)
        encoding_network_num_train_steps = kwargs.get('encoding_network_num_train_steps', 10)

        encoder = NeuralLinUCBLearner.SimpleNet(self.observation_spec, encoding_dim=encoding_dim, hidden_neurons=hidden_neurons)
        self.agent = neural_linucb_agent.NeuralLinUCBAgent(
            time_step_spec=self.time_step_spec,
            action_spec=self.action_spec,
            alpha=alpha,
            gamma=gamma,
            encoding_network=encoder,
            encoding_network_num_train_steps=encoding_network_num_train_steps,
            encoding_dim=encoder._encoding_dim,
            optimizer=tf.keras.optimizers.Adam())
