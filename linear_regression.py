import numpy as np

def compute_error(w, b, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w * x + b)) ** 2

    return totalError / len(points)

def step_gradient(w_current, b_current, points, learningRate):
    w_gradient = 0
    b_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        w_gradient += (2/N)*x*(w_current * x + b_current - y)
        b_gradient += (2/N)*(w_current*x + b_current - y)

    new_w = w_current - learningRate * w_gradient
    new_b = b_current - learningRate * b_current
    return (new_w, new_b)

def gradient_descent_runner(points, starting_b, starting_w, learning_rate, num_iterations):
    w = starting_w
    b = starting_b
    for i in range(num_iterations):
        (w, b) = step_gradient(w, b, np.array(points), learning_rate)
    return (w, b)

def run():
    points = np.genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0  # initial y-intercept guess
    initial_w = 0  # initial slope guess
    num_iterations = 1000
    print("Starting gradient descent at b = {0}, w = {1}, error = {2}"
          .format(initial_b, initial_w,
                  compute_error(initial_b, initial_w, points))
          )
    print("Running...")
    [w, b] = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, w = {2}, error = {3}".
          format(num_iterations, b, w,
                 compute_error(w, b, points))
          )


if __name__ == '__main__':
    run()
