# Copyright 2019 (c) Aalto University - All Rights Reserved
# ELEC-E8125 - Reinforcement Learning Course
# AALTO UNIVERSITY
#
#############################################################
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)
def normalize_angle(x):
    # Hint: Use the angle normalize for normalize theta
    return ((x+np.pi) % (2*np.pi)) - np.pi

def reshape_state(x):

    theta = np.arctan2(x[1], x[0])
    
    if (theta >= 0):
        theta = theta - 2 * np.pi
    #TASK-2    
    theta=normalize_angle(theta)
    return np.array((theta, x[2]))

def perform_rollouts(env, timesteps, render):

    # Define a new empty observations list
    state_action_vec = []
    targets_vec = []

    state = env.reset()  # Get the initial state
    state = reshape_state(state)
    # Take a new observation in the environment
    for t in range(timesteps):

        if render:
            env.render()

        u_mean = np.random.uniform(env.action_space.low, env.action_space.high)

        state_new, _, done, _ = env.step(u_mean)  # Take a new action in the environment
        state_new = reshape_state(state_new)
        state_action_vec.append(np.hstack((state, u_mean)))  # Stack the observed state and action taken

        # Append the new target
        target = state_new - state
        targets_vec.append(target)

        state = state_new

        if done:
            break

    inputs = np.stack(state_action_vec)
    targets = np.stack(targets_vec)

    return inputs, targets


def plot_cost(cost):
    plt.figure(1)
    plt.plot(cost)
    plt.title('Trajectory Cost...')
    plt.xlabel('Timestep')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.show()
