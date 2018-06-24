import numpy as np


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


class Network:
    # Adapted from https://medium.com/@jamesloyys/here-you-go-9b3d91e2202e
    def __init__(self, x, y, dims):
        self.input = x
        self.y = y
        self.dims = dims
        self.weights = np.array([np.random.rand(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.layers = [self.input] + [np.zeros(dims[i]) for i in range(1, len(dims))]
        self.output = self.layers[-1]

    def feedforward(self):
        h_layer = self.input
        for i in range(len(self.dims) - 1):
            # x2 = s(w1*x1 + b1)
            h_layer = sigmoid(np.dot(h_layer, self.weights[i]))
            self.layers[i + 1] = h_layer

        self.output = self.layers[-1]

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights.
        d_weights = [0] * (len(self.layers) - 1)
        d_out = 2 * (self.y - self.layers[-1]) * sigmoid_derivative(self.layers[-1])
        d_weights[-1] = np.dot(self.layers[-2].T, d_out)

        for i in range(len(self.layers) - 2, 0, -1):
            d_weights[i - 1] = np.dot(self.layers[i - 1].T,
                                      np.dot(d_out, self.weights[i].T) * sigmoid_derivative(self.layers[i]))

        for w, dw in zip(self.weights, d_weights):
            w += dw


class CustomNetwork:
    # Feed-forward only Network
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

    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    y = np.array([[0], [1], [1], [0]])

    nn = Network(X, y, [3, 4, 1])

    for i in range(10000):
        nn.feedforward()
        nn.backprop()

    print(nn.output)

    # Using pre-generated weights
    # dims = [3, 4, 4, 3]
    # weights = np.array([np.random.rand(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
    # net = CustomNetwork(weights, dims)
    #
    # for _ in range(10):
    #     x = np.random.rand(3)
    #     print(x, net.feedforward(x))
