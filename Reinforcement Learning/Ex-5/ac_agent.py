import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from utils import discount_rewards

class Value(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2 = torch.nn.Linear(self.hidden, action_space)
        self.fc3 = torch.nn.Linear(action_space,1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x,episode_number=0):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2_mean = torch.nn.Linear(self.hidden, action_space)
        self.fc3 = torch.nn.Linear(self.hidden,1)
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
        return Normal(mu, sigma),self.fc3(x)

        # TODO: Add a layer for state value calculation (T3)


class Agent(object):
    def __init__(self, policy,value):
        self.train_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.policy = policy.to(self.train_device)
        self.values_fn = value.to(self.train_device)
        self.optimizer_p = torch.optim.RMSprop(policy.parameters(), lr=5e-3)
        #self.optimizer_v = torch.optim.RMSprop(value.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.states = []
        self.action_probs = []
        self.rewards = []
        self.all_dones=[]
        self.values=[]

    def episode_finished(self, end_state_value,episode_number):
        action_probs = torch.stack(self.action_probs, dim=0) \
                .to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        all_values = torch.stack(self.values, dim=0).to(self.train_device).squeeze(-1)
        dones = torch.tensor(self.all_dones, dtype=torch.bool)
        self.states, self.action_probs, self.rewards,self.values,self.all_dones = [], [], [],[],[]

        # TODO: Compute discounted rewards (use the discount_rewards function)
        #We need the next_value with the terminal_value
        next_values = torch.cat((all_values[1:], end_state_value))
        next_values[dones == True] = 0
    
        delta = rewards + self.gamma * next_values - all_values

        # TODO: Compute critic loss and advantages (T3)
        
        p_loss = (- action_probs * delta.detach()).mean()
        c_loss = (- all_values * delta.detach()).mean()
        loss=p_loss+c_loss
        loss.backward()
     
        
        self.optimizer_p.step()
        self.optimizer_p.zero_grad()
        
        
        # TODO: Compute the gradients of loss w.r.t. network parameters (T1)

        # TODO: Update network parameters using self.optimizer and zero gradients (T1)


    def get_action(self, observation, evaluation=False,episode_number=0):
        x = torch.from_numpy(observation).float().to(self.train_device)

        # TODO: Pass state x through the policy network (T1)
        normal_dist,value_new=self.policy.forward(x,episode_number)
        # TODO: Return mean if evaluation, else sample from the distribution
        # returned by the policy (T1)
        
        if  evaluation:
            action= normal_dist.mean
        else:
           action = normal_dist.sample()
        # TODO: Calculate the log probability of the action (T1)
        act_log_prob = normal_dist.log_prob(action) 
        
        # TODO: Return state value prediction, and/or save it somewhere (T3)


        return action, act_log_prob,value_new

    def store_outcome(self, observation, action_prob, action_taken, reward,value,done):
        self.states.append(observation)
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.values.append(value)
        self.all_dones.append(done)
