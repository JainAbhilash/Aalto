import numpy as np
import torch
import matplotlib.pyplot as plt


def some_linear_function(x):
    return 3*x + 2


class GradientRegressor():
    def __init__(self, lr=1e-2):
        self.lr = lr
        self.a, self.b = None, None
        self.a_hist, self.b_hist, self.loss_hist = [], [], []

    def fit(self, xs, ys, niter=100):
        self.a = torch.randn(1, requires_grad=True)
        self.b = torch.randn(1, requires_grad=True)
        optimizer = torch.optim.SGD([self.a, self.b], lr=self.lr)
        loss = 0
        for it in range(niter):
            # Mean square error (MSE)
            # x**n => x to the power of n (like x^n in Matlab)
            loss = torch.mean((self.a * xs + self.b - ys)**2)

            # Store data in list
            self.a_hist.append(self.a.item())
            self.b_hist.append(self.b.item())
            self.loss_hist.append(loss)

            # Gradients of loss w.r.t. a and b
            loss.backward()

            # Updated values
            optimizer.step()
            optimizer.zero_grad()
        # The 'print' function prints text to the console
        # The % sign is used for string formatting
        # (%.3f - floating point number, 3 decimal places)
        print("Finished! a=%.3f, b=%.3f, loss=%.3f" % (self.a.item(), self.b.item(), loss.item()))

    def eval(self, xs):
        return self.a * xs + self.b


class AnalyticalRegressor():
    def __init__(self):
        self.a = None
        self.b = None

    def fit(self, xs, ys):
        # Compute the analytical least squares solution using pseudoinverse
        # Number of input points
        npoints = xs.shape[0]
        # Create an array of ones, shaped (n, 2)
        xmat = torch.ones((npoints, 2))

        # Fill in the first column with x values
        xmat[:, 0] = xs

        # Compute the pseudoinverse
        x_pinv = torch.pinverse(xmat)

        # Calculate the [a, b] vector ( @ - matrix multiplication)
        theta = x_pinv @ ys

        # Read the values from the result vector and store them
        self.a = theta[0]
        self.b = theta[1]
        print("Analytical solution: a=%.3f, b=%.3f" % (self.a, self.b))

    def eval(self, xs):
        return self.a * xs + self.b


def generate_points(function, x_range=(-10, 10), amount=100, noise=0):
    xs = torch.linspace(x_range[0], x_range[1], amount)
    ys = function(xs) + noise*torch.randn(*xs.shape)
    return xs, ys


if __name__ == "__main__":
    # Generate data
    xs, ys = generate_points(some_linear_function, noise=3)

    # Create and fit gradient regressor
    regressor = GradientRegressor()
    regressor.fit(xs, ys)

    # Create analytical regressor for comparison
    an_regressor = AnalyticalRegressor()
    an_regressor.fit(xs, ys)

    # Plot training progress
    plt.plot(regressor.a_hist)

    # Subsequent calls to plot use the same plot until the figure is shown
    # on screen (plt.plot) or until it is cleared (plt.clf) - 'hold on' in Matlab
    plt.plot(regressor.b_hist)

    # Add some title, grid and legend to the plot
    plt.title("a and b during training")
    plt.grid(True)
    plt.legend(["a", "b"])

    # Display the plot on the screen
    plt.show()

    # Generate some data
    xs_test = torch.linspace(-10, 10, 30)
    ys_grad = regressor.eval(xs_test)
    ys_anal = an_regressor.eval(xs_test)
    ys_true = some_linear_function(xs_test)

    # Convert things back to NumPy for plotting
    xs = xs.numpy()
    xs_test = xs_test.numpy()
    ys = ys.numpy()
    ys_grad = ys_grad.detach().numpy()
    ys_anal = ys_anal.numpy()
    ys_true = ys_true.numpy()

    # Plot input, estimation result and true values
    plt.plot(xs, ys)
    plt.plot(xs_test, ys_grad)
    plt.plot(xs_test, ys_anal)
    plt.plot(xs_test, ys_true)
    plt.title("Estimation results")
    plt.xlabel("xs")
    plt.ylabel("ys")
    plt.legend(["Data", "Gradient estimate", "Analytical estimate", "True"])
    plt.grid(True)
    plt.show()

