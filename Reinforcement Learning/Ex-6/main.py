# Copyright 2019 (c) Aalto University - All Rights Reserved
# ELEC-E8125 - Reinforcement Learning Course
# AALTO UNIVERSITY
#
#############################################################
import gym
import pickle
import warnings
import autograd.numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge

# Avoid Future and Deprecated warnings when performing predictions
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from iLQR import iLQR
from utils import perform_rollouts, reshape_state, normalize_angle, plot_cost

np.random.seed(7)


# TODO: Task2 Define here your cost function.
def cost_function(x, u):
    Q=np.array([[5,0],[0,1]])
    R=1
    costs=x.T@Q@x + u.T@u
    return costs

# Dynamics used in Task4
def dynamics(x, u):

    theta = x[0]
    dtheta = x[1]

    g = 9.81
    m = 1.0
    l = 1.0
    dt = 0.05
    max_dtheta = 8
    max_u = 2

    u = np.clip(u, -max_u, max_u)[0]

    dtheta_prime = dtheta + (-3 * g / (2 * l) * np.sin(theta + np.pi) + 3. / (m * l ** 2) * u) * dt
    theta_prime = theta + dtheta_prime * dt
    dtheta_prime = np.clip(dtheta_prime, -max_dtheta, max_dtheta)

    x = np.array([theta_prime, dtheta_prime])
    return x


def main(t_horizon):

    # Set the prediction horizon
    ilqr_horizon = 15

    env = gym.make("Pendulum-v0")
    env.seed(7)
    env.reset()

    # Assume start high enough to stabilise
    state, _, _, _ = env.step([0])
    u_traj = [np.array((0.0,)) for _ in range(ilqr_horizon)]

    x_list = []
    state = reshape_state(state)
    x_list.append(state)

    # Create the Kernel Regressor model
    alpha = np.linspace(0.01, 0.1, 10)  # good condition
    gamma = np.linspace(0.5, 2, 10)
    model = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
                         param_grid={"alpha": alpha, "gamma": gamma},
                         scoring='neg_mean_squared_error', return_train_score=True)

    # Perform initial observations with a random policy
    obs_inputs, obs_targets = perform_rollouts(env, t_horizon,
                                               render=False)
    # Perform the model learning
    model.fit(obs_inputs, obs_targets)

    # Create the controller
    controller = iLQR(lambda x, u: cost_function(x, u), 2, 1, dynamics, horizon=ilqr_horizon)

    # Initialize the cost
    cost_list = []

    # Count when we performed the last fit of the model
    last_episode_fit = 0

    # Set the iLQR alpha
    alpha_LQR = 0.99

    for t in range(t_horizon):

        x_traj = [state]
        cost_list.append(cost_function(x_traj[0], u_traj[0]))

        # Perform predictions using the approximated model
        for j in range(controller.horizon):
            observations = np.array([np.append(x_traj[j], u_traj[j])])
            # Comment for Task 4
            # prediction = x_traj[j] + model.predict(observations)
            # x_traj.append(prediction[0])
            x_traj.append(dynamics(x_traj[j], u_traj[j]))
            # TODO Task 4: use x_traj[j].append(dynamics(x_traj[j], u_traj[j])) to compute the next state of the trajectory

        # Perform the iLQR steps
        for rep in range(3):
            # TODO: Task 3 Update the gradients
            # TODO: Task 3 Perform the backward pass
            # TODO: Task 3 Perform the forward pass, use alpha=0.99
            controller.update_gradients(x_traj, u_traj)
            k_traj, K_traj = controller.backward(x_traj, u_traj)
            x_traj, u_traj = controller.forward(model, x_traj, u_traj, k_traj, K_traj, alpha_LQR)
            pass

        x_list.append(state)

        env.render()
        state_new, _, _, _ = env.step(u_traj[0])
        state_new = reshape_state(state_new)

        # Add the new observations to the model when needed
        if t - last_episode_fit > 10 and t < 50:
            new_obs = np.array([np.hstack((state_new, u_traj[0]))])
            new_target = np.array([state_new - state])
            obs_inputs = np.concatenate((obs_inputs, new_obs))
            obs_targets = np.concatenate((obs_targets, new_target))
            # Fit with the new observations and targets
            model.fit(obs_inputs, obs_targets)
            last_episode_fit = t

        # Update the current state
        state = state_new

        print("Timestep ", t, " action:= ", u_traj[0], " cost:= ", cost_function(x_traj[0], u_traj[0]))

    # Once the timesteps are finished, plot the cost
    plot_cost(np.array(cost_list))

    # Save the state trajectory
    with open("xlist.npy", "wb") as f:
        pickle.dump(x_list, f)


if __name__ == "__main__":

    # Set the trajectory timesteps
    trajectory_steps = 500
    main(trajectory_steps)

    exit(2)
