import mnist
import scipy.misc

if __name__ == "__main__":
    images = mnist.train_images()
    scipy.misc.toimage(scipy.misc.imresize(images[0, :, :] * -1 + 256, 10.))
    x = images.reshape(images.shape[0], images.shape[1] * images.shape[2])

    train_images = mnist.train_images()
    train_labels = mnist.train_labels()

    test_images = mnist.test_images()
    test_labels = mnist.test_labels()

    print(test_images[0])
