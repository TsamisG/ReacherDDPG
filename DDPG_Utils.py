import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class OU_Noise():
    def __init__(self, mu, x0, theta=0.2, sigma=0.15, dt=1e-2):
        self.mu = mu
        self.dt = dt
        self.theta = theta
        self.sigma = sigma
        self.x = x0

    def __call__(self):
        dW = np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x = self.x + self.theta * (self.mu - self.x) + self.sigma * dW
        return self.x


class Memory():
  def __init__(self, memory_size, input_shape, n_actions):
    self.memory_size = memory_size
    self.input_shape = input_shape
    self.n_actions = n_actions
    self.counter = 0
    self.state_memory = np.zeros((self.memory_size, *input_shape), dtype=np.float32)
    self.action_memory = np.zeros((self.memory_size, self.n_actions), dtype=np.float32)
    self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
    self.state2_memory = np.zeros((self.memory_size, *input_shape), dtype=np.float32)
    self.done_memory = np.zeros(self.memory_size, dtype=np.int8)

  def remember(self, s, a, r, s2, done):
    idx = self.counter % self.memory_size
    self.state_memory[idx] = s
    self.action_memory[idx] = a
    self.reward_memory[idx] = r
    self.state2_memory[idx] = s2
    self.done_memory[idx] = done
    self.counter += 1

  def sample_batch(self, batch_size):
    current_memory_size = min(self.counter, self.memory_size)
    batch_indices = np.random.choice(current_memory_size, batch_size, replace = False)
    states = self.state_memory[batch_indices]
    actions = self.action_memory[batch_indices]
    rewards = self.reward_memory[batch_indices]
    states2 = self.state2_memory[batch_indices]
    dones = self.done_memory[batch_indices]

    return states, actions, rewards, states2, dones



class CriticNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions, learning_rate=1e-4):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.n_actions+self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        fc1 = F.relu(self.fc1(state))
        fc2 = F.relu(self.fc2(T.cat((action, fc1), dim=1)))
        q = self.q(fc2)
        return q





class ActorNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions, learning_rate=1e-4):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        fc1 = F.relu(self.fc1(state))
        fc2 = F.relu(self.fc2(fc1))
        mu = F.tanh(self.mu(fc2))
        
        return mu




class Agent:
    def __init__(self, lr_actor, lr_critic, input_dims, n_actions, tau, env, gamma=0.99, memory_size=10000,
                 layer1_size=400, layer2_size=300, batch_size=128):

        self.gamma = gamma

        self.action_space_mean = (env.action_space.high + env.action_space.low)/2
        self.action_space_amp = env.action_space.high - self.action_space_mean
        
        self.tau = tau
        self.memory = Memory(memory_size, input_dims, n_actions)
        self.batch_size = batch_size

        self.actor = ActorNetwork(input_dims, layer1_size, layer2_size, n_actions=n_actions, learning_rate=lr_actor)
        self.target_actor = ActorNetwork(input_dims, layer1_size, layer2_size, n_actions=n_actions, learning_rate=lr_actor)

        self.critic = CriticNetwork(input_dims, layer1_size, layer2_size, n_actions=n_actions, learning_rate=lr_critic)
        self.target_critic = CriticNetwork(input_dims, layer1_size, layer2_size, n_actions=n_actions, learning_rate=lr_critic)

        self.noise = OU_Noise(mu=np.zeros(n_actions), x0=np.zeros(n_actions))

        self.initialize_target_networks()

    def choose_action(self, observation, with_noise=True):
        observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor(observation).to(self.actor.device)
        mu_prime = mu + with_noise * T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
        mu_prime = T.clamp(mu_prime, min=-1.0, max=1.0)
        return self.action_space_mean + mu_prime.cpu().detach().numpy() * self.action_space_amp
        #return mu_prime.cpu().detach().numpy()
    
    def remember(self, s, a, r, s2, done):
        self.memory.remember(s, a, r, s2, done)

    def learn(self):
        if self.memory.counter < self.batch_size:
            return

        state, action, reward, state2, done = self.memory.sample_batch(self.batch_size)
        state = T.tensor(state, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        state2 = T.tensor(state2, dtype=T.float).to(self.critic.device)
        done = T.tensor(done).to(self.critic.device)

        target_actions = self.target_actor.forward(state2)
        critic_value2 = self.target_critic.forward(state2, target_actions)
        critic_value = self.critic.forward(state, action)

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value2[j] * (1 - done[j]))

        target = T.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)


        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        actor_loss = - self.critic.forward(state, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_target_networks()

    def initialize_target_networks(self):
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        
        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
        
    
    def update_target_networks(self):
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = self.tau*critic_state_dict[name].clone() + (1-self.tau)*target_critic_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = self.tau*actor_state_dict[name].clone() + (1-self.tau)*target_actor_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)