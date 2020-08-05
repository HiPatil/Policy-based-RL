import numpy as np
import time
import random
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


env = gym.make('InvertedPendulum-v2')

# import mujoco_py


test_env = False
if test_env:
	env.reset()
	for _ in range(1000):
		env.render()
		env.step(env.action_space.sample()) 	# take random actions
	env.close()

print('Observation Shape:', env.observation_space.shape, '\nAction Shape:', env.action_space)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Code is running on:", device)

############ PARAMETERS ####################

N_OBS = env.observation_space.shape[0]
N_ACT = env.action_space.shape[0]
N_EPISODE = 1500
LEARNING_RATE = 0.001
DISCOUNT = 0.99

############### Network for A2C ####################3
class ACNet(nn.Module):
	def __init__(self, observations, actions):
		super(ACNet, self).__init__()
		self.actor = nn.Sequential(
			nn.Linear(observations,  32),
			nn.ReLU(),
			nn.Linear(32, 16),
			nn.ReLU()
			)
		self.mu = nn.Linear(16, actions)
		self.sigma = nn.Linear(16, actions)
		

		self.critic = nn.Sequential(
            nn.Linear(observations, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
            )

	def forward(self, x):
		act = self.actor(x)
		mean = self.mu(act)
		std = F.softplus(self.sigma(act))
		value = self.critic(x)
		# value = self.value(crt)

		return mean, std, value

ac_network = ACNet(N_OBS, N_ACT).to(device)
optimizer = optim.Adam(ac_network.parameters())

writer = SummaryWriter('run/using_tensorboard')

def choose_action(state):
	mu, sigma, value = ac_network(state)
	m = torch.distributions.Normal(mu, sigma)
	action = m.sample()
	log_prob = m.log_prob(action)
	return action.detach().cpu().numpy(), log_prob, value

def compute_returns(next_state, rewards, done, discount = DISCOUNT):
	next_state = torch.FloatTensor(next_state).to(device)
	_, _, next_q_val = ac_network(next_state)
	returns = []
	for step in reversed(range(len(rewards))):
		next_q_val = rewards[step] + discount*next_q_val*(1-done[step])
		returns.append(next_q_val)
	returns.reverse()
	return returns

def ACupdate(log_probs, q_vals, values):
	optimizer.zero_grad()
	ac_loss = 0
	advantage = q_vals - values
#     print(-(log_probs*advantage).sum())
	actor_loss = -(log_probs*advantage.detach()).mean()
	critic_loss = advantage.pow(2).mean()
	
	ac_loss = actor_loss+critic_loss
	ac_loss.backward()
	
	optimizer.step()

	return ac_loss.item()

for i in range(1, N_EPISODE+1):
	ep_rewards = []
	log_probs = []
	done_states = []

	total_reward = 0
	done = False
	values = []
	state = env.reset()
	while not done:
		state = torch.FloatTensor(state).to(device)
		action, log_prob, value = choose_action(state)

		next_state, reward, done, info = env.step(action)

		done = torch.tensor([done], dtype = torch.float, device = device)
		ep_rewards.append(torch.tensor([reward], dtype = torch.float, device = device))
		log_probs.append(log_prob)
		done_states.append(done)
		values.append(value)

		total_reward += reward
		state = next_state

	q_vals = compute_returns(next_state, ep_rewards, done_states)
	q_vals = torch.stack(q_vals)
	values = torch.stack(values)
	log_probs = torch.stack(log_probs)

	loss = ACupdate(log_probs, q_vals, values)

	writer.add_scalar('Attr/Training loss', loss, i)
	writer.add_scalar('Attr/Episode reward', total_reward, i)
	print('Episode Trained:', i)
	
	if i%1000 == 0:
		torch.save(ac_network.state_dict(), 'Models/ACNet_'+str(i)+'.pth')
		print('Model Saved')

print('Done Training')

