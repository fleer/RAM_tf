import tensorflow as tf
import numpy as np

class MNIST():
    """
    Class for downloading and preprocessing the MNIST image dataset
    The Code for creating the glimpses is based on https://github.com/jlindsey15/RAM
    """

    def __init__(self, mnist_size, batch_size, translate, translated_mnist_size, monte_carlo_samples):

        self.mnist_size = mnist_size
        self.batch_size = batch_size
        self.dataset = tf.contrib.learn.datasets.load_dataset("mnist")

        self.translate = translate
        if translate:
            self.translated_mnist_size = mnist_size
            self.mnist_size = translated_mnist_size

        self.M = monte_carlo_samples

    def get_batch(self, batch_size, data_type="train"):
        if data_type == "train":
            X, Y = self.dataset.train.next_batch(batch_size)
        elif data_type == "validation":
            X, Y = self.dataset.validation.next_batch(batch_size)
        elif data_type == "test":
            X, Y = self.dataset.test.next_batch(batch_size)
        else:
            print("Wrong data_type: " + str(data_type) + "!")
            return 0
        if self.translate:
           X, _ = self.convertTranslated(X, self.translated_mnist_size, self.mnist_size, batch_size)
        # duplicate M times, to get M Monte-Carlo Samples of the location during forward pass
        # -----------------------------------
        # Ba, J. L., Mnih, V., Deepmind, G., & Kavukcuoglu, K. (n.d.).
        # MULTIPLE OBJECT RECOGNITION WITH VISUAL ATTENTION.
        # Retrieved from https://arxiv.org/pdf/1412.7755.pdf
        # -----------------------------------
        # See Eq. (9), (10)
        # As the the location prediction is stochastic, the attention model can be
        # evaluated multiple times on the same sample.
        X = np.tile(X, [self.M, 1])
        Y = np.tile(Y, [self.M])
        X = np.reshape(X, (self.batch_size*self.M, self.mnist_size, self.mnist_size, 1))
        return X,Y

    def convertTranslated(self, images, initImgSize, finalImgSize, batch_size):
        size_diff = finalImgSize - initImgSize
        newimages = np.zeros([batch_size, finalImgSize*finalImgSize])
        imgCoord = np.zeros([batch_size,2])
        for k in range(batch_size):
            image = images[k, :]
            image = np.reshape(image, (initImgSize, initImgSize))
            # generate and save random coordinates
            randX = np.random.randint(0, size_diff)
            randY = np.random.randint(0, size_diff)
            imgCoord[k,:] = np.array([randX, randY])
            # padding
            image = np.lib.pad(image, ((randX, size_diff - randX), (randY, size_diff - randY)), 'constant', constant_values = (0))
            newimages[k, :] = np.reshape(image, (finalImgSize*finalImgSize))

        return newimages, imgCoord

