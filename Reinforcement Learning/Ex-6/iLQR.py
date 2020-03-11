# Copyright 2019 (c) Aalto University - All Rights Reserved
# ELEC-E8125 - Reinforcement Learning Course
# AALTO UNIVERSITY
#
#############################################################
import autograd.numpy as np
from autograd import jacobian
from utils import normalize_angle

# Avoid Future and Deprecated warnings when performing predictions
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class iLQR:

    def __init__(self, cost_function, state_dim, action_dim, dynamics, horizon=50):

        self.f = dynamics
        self.horizon = horizon
        self.s_dim = state_dim
        self.a_dim = action_dim

        self.fx_grad = jacobian(self.f, 0)
        self.fu_grad = jacobian(self.f, 1)

        self.C = cost_function

        self.v = [0.0 for _ in range(horizon + 1)]
        self.Vx = [np.zeros(state_dim) for _ in range(horizon + 1)]
        self.Vxx = [np.zeros((state_dim, state_dim)) for _ in range(horizon + 1)]

        self.Fx = np.zeros((self.horizon + 1, self.s_dim, self.s_dim))
        self.Fu = np.zeros((self.horizon + 1, self.s_dim, self.a_dim))
        self.Cx = np.zeros((self.horizon + 1, self.s_dim))
        self.Cu = np.zeros((self.horizon + 1, self.a_dim))
        self.Cxx = np.zeros((self.horizon + 1, self.s_dim, self.s_dim))
        self.Cuu = np.zeros((self.horizon + 1, self.a_dim, self.a_dim))
        self.Cux = np.zeros((self.horizon + 1,self.a_dim, self.s_dim))  # For the cart-pole problem this is kept as zeros

    def update_gradients(self, x_traj, u_traj):

        for t in range(self.horizon):
            x = x_traj[t]
            u = u_traj[t]

            # Compute the gradients of the dynamics
            self.Fx[t] = self.fx_grad(x, u)
            self.Fu[t] = self.fu_grad(x, u)

            # TODO: Compute here the cost gradient for your cost function
            self.Cx[t] = self.cost_dx(x,u)
            self.Cu[t] = self.cost_du(x,u)
            # TODO: Compuete here the hessian for your cost function
            self.Cxx[t] = self.cost_dxx(x,u)
            self.Cuu[t] = self.cost_duu(x,u)

            self.Cux[t] = self.cost_dux(x,u)

    def backward(self, x_traj, u_traj):
        self.v[-1] = self.C(x_traj[-1], u_traj[-1])
        self.Vx[-1] = self.cost_dx(x_traj[-1], u_traj[-1])
        self.Vxx[-1] = self.cost_dxx(x_traj[-1], u_traj[-1])

        k_traj = []
        K_traj = []

        for t in range(self.horizon - 1, -1, -1):
            Vx = self.Vx[t + 1]
            Vxx = self.Vxx[t + 1]

            # Compute the action-value function
            Qx = self.Cx[t] + np.matmul(self.Fx[t].T, Vx)
            Qu = self.Cu[t] + np.matmul(self.Fu[t].T, Vx)

            # Simplified action-value function
            Qxx = self.Cxx[t] + np.matmul(np.matmul(self.Fx[t].T, Vxx),
                                          self.Fx[t])
            Quu = self.Cuu[t] + np.matmul(np.matmul(self.Fu[t].T, Vxx),
                                          self.Fu[t])
            Qux = self.Cux[t] + ((self.Fu[t].T).dot(Vxx)).dot(self.Fx[t])

            # Compute the inverse
            Quu_inv = np.linalg.inv(Quu.reshape(1, 1) + 1e-9 * np.eye(Quu.shape[0]))

            k = -Quu_inv.dot(Qu)  # Open-loop gain
            K = -Quu_inv.dot(Qux)  # Closed-loop gain

            # Compute the value function
            self.v[t] += 0.5 * (k.T.dot(Quu)).dot(k) + k.T.dot(Qu)
            self.Vx[t] = Qx + (K.T.dot(Quu)).dot(k) + K.T.dot(Qu) + Qux.T.dot(k)
            self.Vxx[t] = Qxx + (K.T.dot(Quu)).dot(K) + K.T.dot(Qux) + Qux.T.dot(K)

            k_traj.append(k)
            K_traj.append(K)

        k_traj.reverse()
        K_traj.reverse()

        return k_traj, K_traj

    def forward(self, model, x_traj, u_traj, k_traj, K_traj, alpha):

        x_traj_new = np.array(x_traj)
        u_traj_new = np.array(u_traj)

        for t in range(len(u_traj)):
            control = alpha ** t * k_traj[t] + np.matmul(K_traj[t], (x_traj_new[t] - x_traj[t]))
            u_traj_new[t] = np.clip(u_traj[t] + control, -2, 2)

            # Create the vector of observations X={x,u}
            observations = np.array([np.append(x_traj_new[t], u_traj_new[t])])
            # Comment for Task 4
            #test_predict = x_traj_new[t] + model.predict(observations)
            # TODO Task 4: use the dynamics of the system instead of the learned model compute the next state
            #x_traj_new[t + 1] = test_predict
            x_traj_new[t+1]=self.f(x_traj_new[t],u_traj_new[t])

        return x_traj_new, u_traj_new
    
    # TODO: Task2 Define here the gradient of your cost function respect to x
    def cost_dx(self, x, u):
        Q=np.array([[5,0],[0,1]])
        dx = 2*Q@x
        return dx

    # TODO: Task2 Define here the hessian of your cost function respect to x
    def cost_dxx(self, x, u):
        Q=np.array([[5,0],[0,1]])
        dxx = 2*Q
        return dxx

    # TODO: Task2 Define here the gradient of your cost function respect to u
    def cost_du(self, x, u):
        du = 2*u
        return du

    # TODO: Task2 Define here the Hessian of your cost function respect to u
    def cost_duu(self, x, u):
        #As R is 1
        duu = 2*1
        return duu

    def cost_dux(self, x, u):
        dux=0
        return dux
