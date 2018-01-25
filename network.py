import random
from math import e


def sigmoid(a):
    return 1 / (1 + e ** -a)


# Mean Square Error
def MS_sum_loss(desired, actual):
    return sum(map(lambda x, y: (x - y) ** 2, desired, actual))


def MS_loss(desired, actual):
    return list(map(lambda x, y: (x - y) ** 2, desired, actual))


def diff_loss(desired, actual):
    return list(map(lambda x, y: x - y, desired, actual))


class Node:
    def __init__(self):
        self.weights = []
        self.activation = 0

    def calculate_activation(self, layer):
        self.activation = sigmoid(sum(w * a for w, a in zip(self.weights, [n.activation for n in layer])))


if __name__ == '__main__':
    learning_rate = 0.01
    inputs = [1, 0, 1, 0, 1]
    desired_outputs = [0, 1, 0, 1, 0]

    layers = [[] for _ in range(4)]
    for L in layers:  # Node creation
        for i in range(5):
            L.append(Node())

    for i in range(1, len(layers)):  # Weight creation
        for n in layers[i]:
            n.weights = [random.uniform(-1, 1) for _ in range(len(layers[i - 1]))]

    for i, n in enumerate(layers[0]):  # Inputs
        n.activation = inputs[i]

    for i in range(1, len(layers)):  # Activation calc
        for n in layers[i]:
            n.calculate_activation(layers[i - 1])

    for L in layers:  # Print activations
        for n in L:
            print(end=str(n.activation) + "\t")
        print()

    loss_values = diff_loss(desired_outputs, [n.activation for n in layers[-1]])
    # overall_loss = MS_sum_loss(desired_outputs, [n.activation for n in layers[-1]])
    print("losses:\t", loss_values)
    # print("overall loss:\t", overall_loss)

    for i in range(len(layers) - 1, 0):
        for n, w in zip(layers[i], loss_values):
            node_loss = diff_loss(w, n.activation)
            sum_node_weights = sum([abs(nw) for nw in w])
            for old_w in n.weights:
                loss = diff_loss()  # Calc loss for weights of each node
