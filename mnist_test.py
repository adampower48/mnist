import mnist
import numpy as np

from network import Network


def label_to_arr(n):
    arr = np.zeros(10)
    arr[n] = 1
    return arr


if __name__ == "__main__":
    images = mnist.train_images()
    # scipy.misc.toimage(scipy.misc.imresize(images[0, :, :] * -1 + 256, 10.))
    x = images.reshape(images.shape[0], images.shape[1] * images.shape[2])

    train_images = mnist.train_images()
    train_labels = mnist.train_labels()

    test_images = mnist.test_images()
    test_labels = mnist.test_labels()

    # print(train_images[0])

    trainI = x[:50]
    print(trainI.shape)
    trainL = np.array(list(map(label_to_arr, train_labels[:50])))
    print(trainL.shape)

    net = Network(trainI, trainL, [784, 4, 10])

    for _ in range(100):
        net.feedforward()
        net.backprop()

    print(net.output)
