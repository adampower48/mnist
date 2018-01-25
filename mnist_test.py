from mnist import MNIST


def squish_image(img):
    return list(map(lambda v: v / 256, img))


if __name__ == "__main__":

    mndata = MNIST("samples")

    images, labels = mndata.load_training()

    img = squish_image(images[0])
    for i in range(len(images[0])):
        if i % 28 == 27:
            print(img[i])
        else:
            print(end=str(img[i]) + "\t")
