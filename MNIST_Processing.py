import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

class MNIST():
    """
    Class for downloading and preprocessing the MNIST image dataset
    The Code for creating the glimpses is based on https://github.com/jlindsey15/RAM
    """

    def __init__(self, mnist_size, batch_size, translate, translated_mnist_size):

        self.mnist_size = mnist_size
        self.batch_size = batch_size
        self.dataset = tf.contrib.learn.datasets.load_dataset("mnist")

        self.translate = translate
        if translate:
            self.translated_mnist_size = mnist_size
            self.mnist_size = translated_mnist_size

    def get_batch_train(self, batch_size):
        X, Y = self.dataset.train.next_batch(batch_size)
        if self.translate:
           X, _ = self.convertTranslated(X, self.translated_mnist_size, self.mnist_size, batch_size)
        return X,Y

    def get_batch_test(self, batch_size):
        X, Y = self.dataset.test.next_batch(batch_size)
        if self.translate:
            X, _ = self.convertTranslated(X, self.translated_mnist_size, self.mnist_size, batch_size)
        return X,Y

    def get_batch_validation(self, batch_size):
        X, Y = self.dataset.validation.next_batch(batch_size)
        if self.translate:
            X, _ = self.convertTranslated(X, self.translated_mnist_size, self.mnist_size, batch_size)
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

