import gym
import numpy as np
from matplotlib import pyplot as plt
import seaborn

np.random.seed(123)

env = gym.make('CartPole-v0')
env.seed(321)

episodes = 20000
test_episodes = 10
num_of_actions = 2

# Reasonable values for Cartpole discretization
discr = 16
x_min, x_max = -2.4, 2.4
v_min, v_max = -3, 3
th_min, th_max = -0.3, 0.3
av_min, av_max = -4, 4

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
#initial_q = 50 
# Create discretization grid
x_grid = np.linspace(x_min, x_max, discr)
v_grid = np.linspace(v_min, v_max, discr)
th_grid = np.linspace(th_min, th_max, discr)
av_grid = np.linspace(av_min, av_max, discr)
grid_values=(x_grid,v_grid,th_grid,av_grid)
q_grid = np.zeros((discr, discr, discr, discr, num_of_actions)) + initial_q

def discretize(state):
    disc_state = []
    for i in range(len(state)):
        disc_state.append(np.abs(grid_values[i] - state[i]).argmin())
    return tuple(disc_state)
# Training loop
ep_lengths, epl_avg = [], []
for ep in range(episodes+test_episodes):
    test = ep > episodes
    state, done, steps = env.reset(), False, 0
    #epsilon = 0.2  # T1: GLIE/constant, T3: Set to 0
    epsilon = 0 #GLIE we start from 1.0
    while not done:
        #epsilon=a/(a+ep)
        # TODO: IMPLEMENT HERE EPSILON-GREEDY
        discrete_state = discretize(state)
        if (np.random.random() < epsilon):
            action = int(np.random.rand()*2)
        else:
            # act_0 = discrete_state=discretize(state)
            # act_1 = q_grid[disc_vals + (1,)]             
            # action = np.array([act_0, act_1]).argmax()
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
        if not done:
            state = new_state
        steps += 1
    ep_lengths.append(steps)
    epl_avg.append(np.mean(ep_lengths[max(0, ep-500):]))
    if ep % 200 == 0:
        print("Episode {}, average timesteps: {:.2f}".format(ep, np.mean(ep_lengths[max(0, ep-200):])))

# Save the Q-value array
# np.save("q_values_GLIE.npy", q_grid)  # TODO: SUBMIT THIS Q_VALUES.NPY ARRAY

# # Calculate the value function
# values = np.zeros(q_grid.shape[:-1])  # TODO: COMPUTE THE VALUE FUNCTION FROM THE Q-GRID
# np.save("value_func_GLIE.npy", values)  # TODO: SUBMIT THIS VALUE_FUNC.NPY ARRAY


# Plot the heatmap
# TODO: Plot the heatmap here using Seaborn or Matplotlib
#seaborn.heatmap(np.vstack(q_grid[0]/np.mean(q_grid[1]),q_grid[2]/np.mean(q_grid[3])).T)
#seaborn.heatmap(np.mean(q_grid,axis=(1,3))[:,:,0])
# values = np.max(q_grid, axis=4)
# seaborn.heatmap(np.mean(values,axis=(1,3)))
# plt.savefig('heat_glie.png')
# plt.show()
# Draw plots
plt.plot(ep_lengths)
plt.plot(epl_avg)
plt.legend(["Episode length", "500 episode average"])
plt.title("Episode lengths")
plt.savefig('average_0_Task_3.jpeg')
plt.show()

