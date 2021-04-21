import numpy as np

def sigmoid(x):
    z = np.exp(-x)
    return 1 / (1 + z)


def sigmoid_derivative(x):
    z = sigmoid(x)
    return z * (1 - z)


# Neural Network class adapted from jamesloyys github
# Source: https://gist.github.com/jamesloyys/ff7a7bb1540384f709856f9cdcdee70d

class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.y = y
        self.output = np.zeros(self.y.shape)
        hidden_nodes = np.int(np.ceil((self.input.shape[1] + self.y.shape[1]) / 2))
        self.weights1 = np.random.rand(self.input.shape[1], hidden_nodes)
        self.weights2 = np.random.rand(hidden_nodes, self.y.shape[1])

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backpropagation(self):
        d_w2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
        d_w1 = np.dot(self.input.T, (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output),
                                                  self.weights2.T) * sigmoid_derivative(self.layer1)))
        self.weights1 += d_w1
        self.weights2 += d_w2

    def predict(self, x):
        hidden = sigmoid(np.dot(x, self.weights1))
        return sigmoid(np.dot(hidden, self.weights2))
