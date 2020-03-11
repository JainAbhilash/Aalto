import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from utils import discount_rewards


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2_mean = torch.nn.Linear(self.hidden, action_space)
        # TODO: Implement accordingly (T1, T2)
        #Task1
        #self.sigma=torch.tensor([5],dtype=torch.float32,device=self.device)
        #Task2A
        #self.sigma=torch.tensor([10],dtype=torch.float32,device=self.device)
        #Task2B
        self.sigma=torch.nn.Parameter(torch.tensor([10.]))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x,episode_number=0):
        x = self.fc1(x)
        x = F.relu(x)
        mu = self.fc2_mean(x)
        sigma = self.sigma  # TODO: Is it a good idea to leave it like this?
        #Task 2A
        #sigma = self.sigma*np.e**(-5*10**(-4)*episode_number) 
        # TODO: Instantiate and return a normal distribution
        # with mean mu and std of sigma (T1)
        return Normal(mu, sigma)

        # TODO: Add a layer for state value calculation (T3)


class Agent(object):
    def __init__(self, policy):
        self.train_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.RMSprop(policy.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.states = []
        self.action_probs = []
        self.rewards = []

    def episode_finished(self, episode_number):
        action_probs = torch.stack(self.action_probs, dim=0) \
                .to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        self.states, self.action_probs, self.rewards = [], [], []

        # TODO: Compute discounted rewards (use the discount_rewards function)
        discounted_rewards = discount_rewards(rewards, self.gamma)
        #Task1.3
        discounted_rewards -= torch.mean(discounted_rewards)
        discounted_rewards /= torch.std(discounted_rewards)

        # TODO: Compute critic loss and advantages (T3)
        
        # TODO: Compute the optimization term (T1, T3)
        T = len(rewards)    
        gammas = torch.tensor([self.gamma**t for t in range(T)]).to(self.train_device)
        #baseline=20(Task 1b)
        #optim = -gammas*(discounted_rewards-20)*action_probs
        optim = -gammas*discounted_rewards*action_probs
        loss=optim.sum()
        loss.backward()

        # TODO: Compute the gradients of loss w.r.t. network parameters (T1)

        # TODO: Update network parameters using self.optimizer and zero gradients (T1)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_action(self, observation, evaluation=False,episode_number=0):
        x = torch.from_numpy(observation).float().to(self.train_device)

        # TODO: Pass state x through the policy network (T1)
        normal_dist=self.policy.forward(x,episode_number)
        # TODO: Return mean if evaluation, else sample from the distribution
        # returned by the policy (T1)
        
        if  evaluation:
            action= normal_dist.mean
        else:
           action = normal_dist.sample()
        # TODO: Calculate the log probability of the action (T1)
        act_log_prob = normal_dist.log_prob(action) 
        
        # TODO: Return state value prediction, and/or save it somewhere (T3)


        return action, act_log_prob

    def store_outcome(self, observation, action_prob, action_taken, reward):
        self.states.append(observation)
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))

