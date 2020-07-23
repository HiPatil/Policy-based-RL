import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import optimizers, Model
import gym
import mujoco_py
import numpy as np
import tensorflow_probability as tfp

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)


print(tf.config.list_physical_devices('GPU'))
env = gym.make('InvertedPendulum-v2')
# env.reset()
# for _ in range(100):
# 	env.render()
# 	env.step(env.action_space.sample())
# env.close()

N_ACT = env.action_space.shape[0]
N_OBS = env.observation_space.shape[0]
print(N_OBS, N_ACT)

class reinforce(Model):
	def __init__(self, obs, act):
		super(reinforce, self).__init__()
		
		self.l1 = layers.Dense(units = 32, activation = 'relu')
		self.l2 = layers.Dense(units = 16, activation = 'relu')
		self.l3 = layers.Dense(units = act, activation = None)

	def call(self, x):
		layer1 = self.l1(x)
		layer2 = self.l2(layer1)
		mean = self.l3(layer2)

		return mean

policy = reinforce(N_OBS, N_ACT)
optimizer = optimizers.Adam(lr = 0.01)

def choose_action(state, std = 0.0):
	m = policy(tf.convert_to_tensor(np.expand_dims(state, axis = 0),  dtype=tf.float32))
	dist = tfp.distributions.Normal(loc = m, scale = std)
	action = dist.sample()
	log_prob = dist.log_prob(action)

	return action, log_prob


state = env.reset()
action, log_prob = choose_action(state)
print(action)
