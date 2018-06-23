import numpy as np


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


class Network:
    # https://medium.com/@jamesloyys/here-you-go-9b3d91e2202e
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T, (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output),
                                                  self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2


class CustomNetwork:
    # For use with pre-generated weights
    def __init__(self, weights, dims):
        self.weights = weights
        self.dims = dims

    def feedforward(self, x):
        h_layer = x
        for i in range(len(self.dims) - 1):
            # x2 = s(w1*x1 + b1)
            h_layer = sigmoid(np.dot(h_layer, self.weights[i]))

        return h_layer


if __name__ == '__main__':
    # Standard neural network
    # 1: 0,0,1 -> 0
    # 2: 0,1,1 -> 1
    # 3: 1,0,1 -> 1
    # 4: 1,1,1 -> 0
    #
    # X = np.array([[0, 0, 1],
    #               [0, 1, 1],
    #               [1, 0, 1],
    #               [1, 1, 1]])
    # y = np.array([[0], [1], [1], [0]])
    #
    # nn = Network(X, y)
    #
    # for i in range(10000):
    #     nn.feedforward()
    #     nn.backprop()
    #
    # print(nn.output)

    # Using pre-generated weights
    dims = [3, 4, 4, 3]
    weights = np.array([np.random.rand(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
    net = CustomNetwork(weights, dims)

    for _ in range(10):
        x = np.random.rand(3)
        print(x, net.feedforward(x))
