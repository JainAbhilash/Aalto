import gym
import numpy as np
import matplotlib
matplotlib.use('agg') 
from matplotlib import pyplot as plt
import seaborn

np.random.seed(123)

env = gym.make('LunarLander-v2')
env.seed(321)

episodes = 20000
test_episodes = 10
num_of_actions = 4

# Reasonable values for Cartpole discretization
discr = 16
x_min, x_max = -1.2, 1.2
y_min, y_max = -0.3, 1.2
xdot_min,xdot_max= -2.4,2.4
ydot_min,ydot_max=-2,2
th_min, th_max = -6.28,6.28
thetadot_min, thetadot_max = -8, 8

# For LunarLander, use the following values:
#         [  x     y  xdot ydot theta  thetadot cl  cr
# s_min = [ -1.2  -0.3  -2.4  -2  -6.28  -8       0   0 ]
# s_max = [  1.2   1.2   2.4   2   6.28   8       1   1 ]

# Parameters
gamma = 0.98
alpha = 0.1
target_eps = 0.1
#a = 0  # TODO: Set the correct value.
a=2222
initial_q = 0  # T3: Set to 50

# Create discretization grid
x_grid = np.linspace(x_min, x_max, discr)
y_grid = np.linspace(y_min, y_max, discr)
xdot_grid = np.linspace(xdot_min, xdot_max, discr)
ydot_grid = np.linspace(ydot_min, ydot_max, discr)
th_gird=np.linspace(th_min, th_max, discr)
thetadot_grid=np.linspace(thetadot_min,thetadot_max , discr)
cl_grid=np.linspace(0,1,2)
cr_grid=np.linspace(0,1,2)
grid_values=(x_grid,y_grid,xdot_grid,ydot_grid,th_gird,thetadot_grid,cl_grid,cr_grid)

q_grid = np.zeros((discr,discr,discr, discr, discr, discr,2,2 ,num_of_actions)) + initial_q

def discretize(state):
    disc_state = []
    for i in range(len(state)):
        disc_state.append(np.abs(grid_values[i] - state[i]).argmin())
    return tuple(disc_state)
# Training loop
ep_lengths, epl_avg = [], []
rewards_=[]
for ep in range(episodes+test_episodes):
    test = ep > episodes
    state, done, steps = env.reset(), False, 0
    #epsilon = 0.2  # T1: GLIE/constant, T3: Set to 0
    epsilon = 1.0 #GLIE we start from 1.0
    rewards=0
    while not done:
        epsilon=a/(a+ep)
        # TODO: IMPLEMENT HERE EPSILON-GREEDY
        discrete_state = discretize(state)
        if (np.random.random() < epsilon):
            action = int(np.random.rand()*4)
        else:
            action = np.argmax(q_grid[discrete_state])
        
        new_state, reward, done, _ = env.step(action)
        

        if not test:
            # TODO: ADD HERE YOUR Q_VALUE FUNCTION UPDATE
            state_with_action=(*discretize(state),action)
            Q_curr=q_grid[state_with_action]
            new_action=np.argmax(q_grid[discretize(new_state)])
            new_state_act=(*discretize(new_state),new_action)
            if not done:
                q_next=q_grid[new_state_act]
            else:
                q_next=0
            q_grid[state_with_action]=Q_curr+alpha*(reward+(gamma*q_next)-Q_curr)
            
            
        else:
            #env.render()
            pass
        state = new_state
        steps += 1
        rewards+=reward
    ep_lengths.append(steps)
    rewards_.append(rewards)
    epl_avg.append(np.mean(rewards_[max(0, ep-500):]))
    if ep % 200 == 0:
        print("Episode {}, average timesteps: {:.2f}".format(ep, np.mean(ep_lengths[max(0, ep-200):])))

# Save the Q-value array
#np.save("q_values.npy", q_grid)  # TODO: SUBMIT THIS Q_VALUES.NPY ARRAY

# Calculate the value function
values = np.zeros(q_grid.shape[:-1])  # TODO: COMPUTE THE VALUE FUNCTION FROM THE Q-GRID
#np.save("value_func_GLIE.npy", values)  # TODO: SUBMIT THIS VALUE_FUNC.NPY ARRAY


# Plot the heatmap
# TODO: Plot the heatmap here using Seaborn or Matplotlib
#seaborn.heatmap(np.vstack(q_grid[0]/np.mean(q_grid[1]),q_grid[2]/np.mean(q_grid[3])).T)
#seaborn.heatmap(np.mean(q_grid,axis=(1,3))[:,:,0])
# values = np.max(q_grid, axis=4)
# seaborn.heatmap(np.mean(values,axis=(1,3)))
# plt.savefig('heat_lunar.png')
# plt.show()
# Draw plots
plt.plot(rewards_)
plt.plot(epl_avg)
#plt.legend(["Episode length", "500 episode average"])
plt.legend(["Rewards", "500 reward average"])
#plt.title("Episode lengths")
plt.title("Rewards")

plt.savefig('average_lunar_reward.jpeg')
plt.show()

