# Import statements are used to include libraries in the program
# NumPy is used for all kinds of math (like matrix operations),
# while PyPlot is used for plotting. 'as' can be used to change/shorten the name
# of the imported module (to save typing or resolve naming conflicts)
import numpy as np
import matplotlib.pyplot as plt


# Function definition is done with 'def'
# Indentation indicates beginning/end of a block (like braces {} in C/Java)
def some_linear_function(x):
    return 3*x + 2


# Class definition
# Similarly, the contents are indented
class GradientRegressor():
    # Initializer method (pretty much a constructor)
    # `self` is a reference to the class instance
    # 'lr' has a default value of 1e-2 (or 0.01)
    def __init__(self, lr=1e-2):
        self.lr = lr
        # Initialize a and b predictions with no value
        self.a, self.b = None, None
        # [] creates an empty list
        self.a_hist, self.b_hist, self.loss_hist = [], [], []

    # A function within the class. The first argument points to class instance
    # Conventionally, it's named 'self'
    def fit(self, xs, ys, niter=100):
        # Randomly initialize a and b (a,b ~ N(0, 1))
        self.a = np.random.randn()
        self.b = np.random.randn()
        loss = 0
        # A for loop - range is used to go through all values from 0 to niter-1
        for it in range(niter):
            # Mean square error (MSE)
            # x**n => x to the power of n (like x^n in Matlab)
            loss = np.mean((self.a * xs + self.b - ys)**2)

            # Store data in list
            self.a_hist.append(self.a)
            self.b_hist.append(self.b)
            self.loss_hist.append(loss)

            # Gradients of loss w.r.t. a and b
            dlda = np.mean(2 * xs**2 * self.a + 2*self.b * xs - 2*xs*ys)
            dldb = np.mean(2*xs  * self.a + 2*self.b - 2*ys)

            # Updated values
            self.a = self.a - self.lr * dlda
            self.b = self.b - self.lr * dldb
        # The 'print' function prints text to the console
        # The % sign is used for string formatting
        # (%.3f - floating point number, 3 decimal places)
        print("Finished! a=%.3f, b=%.3f, loss=%.3f" % (self.a, self.b, loss))

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
        xmat = np.ones((npoints, 2))

        # Fill in the first column with x values
        xmat[:, 0] = xs

        # Compute the pseudoinverse
        x_pinv = np.linalg.pinv(xmat)

        # Calculate the [a, b] vector ( @ - matrix multiplication)
        theta = x_pinv @ ys

        # Read the values from the result vector and store them
        self.a = theta[0]
        self.b = theta[1]
        print("Analytical solution: a=%.3f, b=%.3f" % (self.a, self.b))

    def eval(self, xs):
        return self.a * xs + self.b


# Back to the original indentation level - this function DOES NOT belong
# to the class above
def generate_points(function, x_range=(-10, 10), amount=100, noise=0):
    xs = np.linspace(x_range[0], x_range[1], amount)
    # Unary * operator (*xs.shape) "unpacks" a sequence into multiple objects,
    # for example to pass the objects as function arguments. Compare:
    # x = 1, 2, 3  # The variable x holds a sequence ("tuple") of 3 numbers
    # f(x)  # Calls f with a sequence of three numbers, passed as one argument
    # f(*x)  # Calls f with three arguments - the numbers 1, 2, 3
    ys = function(xs) + noise*np.random.randn(*xs.shape)
    return xs, ys


# If condition
# This part of the code will only run if the name of the module is "__main__",
# i.e., when the script is run directly (and not if it is imported
# from another script).
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
    xs_test = np.linspace(-10, 10, 30)
    ys_grad = regressor.eval(xs_test)
    ys_anal = an_regressor.eval(xs_test)
    ys_true = some_linear_function(xs_test)

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

