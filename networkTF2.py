import os

import numpy as np
import tensorflow as tf

def hard_tanh(x):
    """
    Segment-wise linear approximation of tanh
    Faster than standard tanh
    Returns `-1.` if `x < -1.`, `1.` if `x > 1`
    In `-1. <= x <= 1.`, returns `x`
    :param x: A tensor or variable
    :return: A tensor
    """
    lower = tf.convert_to_tensor(-1., x.dtype.base_dtype)
    upper = tf.convert_to_tensor(1., x.dtype.base_dtype)
    x = tf.clip_by_value(x, lower, upper)
    return x

def hard_sigmoid(x):
    """
    Segment-wise linear approximation of sigmoid
    Faster than standard sigmoid
    Returns `0.` if `x < 0.`, `1.` if `x > 1`
    In `0. <= x <= 1.`, returns `x`
    :param x: A tensor or variable
    :return: A tensor
    """
    lower = tf.convert_to_tensor(0., x.dtype.base_dtype)
    upper = tf.convert_to_tensor(1., x.dtype.base_dtype)
    x = tf.clip_by_value(x, lower, upper)
    return x

class AttentionControl(tf.keras.layers.Layer):
    """
    Loop function of recurrent attention network
    :return: next glimpse
    """
    def __init__(self, units, batch_size, pixel_scaling):
        super(AttentionControl, self).__init__()
        # Number of weights
        hg_size = hl_size = 128
        g_size = 256
        self.pixel_scaling = pixel_scaling
        self.batch_size = batch_size 
        self.eval_location_list = []
        self.location_list = []
        self.location_mean_list = []
        self.location_stddev_list = []
        self.glimpses_list = []
        self.baseline_list = []

        # Initialize weights
        self.h_location_std_out = tf.keras.layers.Dense(units,
                kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.1))
        self.b_location_out = tf.keras.layers.Dense(units,
                kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.1))
        self.h_location_out = tf.keras.layers.Dense(units,
                kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.1))
        # Glimpse Net
        self.h_glimpse_layer = tf.keras.layers.Dense(hg_size, activation='relu',
                kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.1))
        self.h_location_layer = tf.keras.layers.Dense(hl_size, activation='relu',
                kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.1))
        self.h_glimpse_layer_sum = tf.keras.layers.Dense(g_size,
                kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.1))
        self.h_location_layer_sum = tf.keras.layers.Dense(g_size,
                kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.1))

    def reset_lists(self):
        self.eval_location_list = []
        self.location_list = []
        self.location_mean_list = []
        self.location_stddev_list = []
        self.glimpses_list = []
        self.baseline_list = []

    def get_lists(self):
        return self.eval_location_list, self.location_list,
        self.location_mean_list, self.location_stddev_list, self.glimpses_list

    def call(self, output, inputs):
        #TODO: First location has to be random
        mean_loc = hard_tanh( self.h_location_out(tf.stop_gradient(output)))
        std_loc = tf.nn.sigmoid( self.h_location_std_out(tf.stop_gradient(output)))
        # Clip location between [-1,1] and adjust its scale
        sample_loc = hard_tanh(mean_loc + tf.cond(self.training,
            lambda: tf.random.normal(mean_loc.get_shape(), 0, std_loc), lambda: 0. ))
        loc = sample_loc * self.pixel_scaling

        # Append stuff to lists
        self.baseline_list.append(self.b_location_out(output))
        self.location_mean_list.append(tf.reduce_sum(mean_loc,1))
        self.location_stddev_list.append(tf.reduce_sum(std_loc,1))
        self.location_list.append(tf.reduce_sum(sample_loc,1))
        self.eval_location_list.append(loc)

        return self.Glimpse_Net(loc, inputs)

    def Glimpse_Net(self, location, inputs):
        """
        Glimpse Network
        :param location:
        :return: Glimpse Network output
        """

        # Get glimpses
        glimpses = self.glimpseSensor(location, inputs)
        # Append glimpses to list for tensorboard summary
        self.glimpses_list.append(glimpses[0])

        # Process glimpses
        glimpses = tf.reshape(glimpses, [self.batch_size, self.totalSensorBandwidth])
        hg = self.h_glimpse_layer(glimpses)

        # Process locations
        hl = self.h_location_layer(location)

        # Combine glimpses and locations
        g = tf.nn.relu(self.h_glimpse_layer_sum(hg) + self.h_location_layer_sum(hl))

        return g

    def glimpseSensor(self, normLoc, inputs):
        """
        Compute Glimpses
        :param normLoc: Location for the next glimpses
        :return: Glimpses
        """
        # Convert location [-1,1] into MNIST Coordinates:
        loc = tf.round(((normLoc + 1.) / 2.) * self.mnist_size)
        loc = tf.cast(loc, tf.int32)

        zooms = []

        # process each image individually
        for k in range(self.batch_size):
            imgZooms = []
            one_img = inputs[k,:,:,:]

            offset = self.sensorBandwidth* (self.scaling ** (self.depth-1))

            # pad image with zeros
            one_img = tf.image.pad_to_bounding_box(one_img, offset, offset, \
                                                   2*offset + self.mnist_size, 2*offset + self.mnist_size)

            # compute the different depth images
            for i in range(self.depth):

                # width/height of the next glimpse
                d = tf.cast(self.sensorBandwidth * (self.scaling ** i), tf.int32)
                r = d//2

                # get mean location
                loc_k = loc[k,:]
                adjusted_loc = offset + loc_k - r

                one_img2 = tf.reshape(one_img, (one_img.get_shape()[0].value, \
                                                one_img.get_shape()[1].value))

                # crop image to (d x d)
                zoom = tf.slice(one_img2, adjusted_loc, [d,d])
                if i > 0:
                    zoom = tf.reshape(zoom, (1, d, d, 1))
                    zoom = tf.image.resize(zoom, (self.sensorBandwidth, self.sensorBandwidth))
                    zoom = tf.reshape(zoom, (self.sensorBandwidth, self.sensorBandwidth))
                imgZooms.append(zoom)
            zooms.append(tf.stack(imgZooms))
        zooms = tf.stack(zooms)
        return zooms

class Decoder(tf.keras.Model):
    def __init__(self, units, batch_size, pixel_scaling):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.units = units
        self.lstm = tf.keras.layers.LSTM(units, activation=tf.nn.relu,
                return_sequences=True, return_state=True,
                recurrent_initializer='zeros')

        # used for attention
        self.attention = AttentionControl(units, pixel_scaling, batch_size)
        print('Decoder Initialized')

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.units))

    def reset_attention(self):
        self.attention.reset_lists()

    def get_attention_lists(self):
        return self.attention.get_lists()

    def call(self, x, hidden, inputs):
        # enc_output shape == (batch_size, max_length, hidden_size)
        glimpse = self.attention(hidden, inputs)

        # passing the concatenated vector to the LSTM
        output, _ = self.lstm(x)

        return output, glimpse 

class RAM():
    """
    Neural Network class, that uses Tensorflow to build and train the Recurrent Attention Model
    """

    def __init__(self, totalSensorBandwidth, batch_size, optimizer, momentum, glimpses, pixel_scaling, mnist_size,
                 channels, scaling_factor, sensorResolution, zooms, lr, lr_decay, lr_decay_steps, lr_decay_type,
                 min_lr):
        """
        Intialize parameters, determine the learning rate decay and build the RAM
        :param totalSensorBandwidth: The length of the networks input vector
                                    ---> nZooms * sensorResolution * sensorResolution * channels
        :param batch_size: Size of each batch
        :param optimizer: The used optimize: "sgd, rmsprop, adadelta, adam,..."
        :param momentum: momentum used by SGD optimizer
        :param glimpses: Number of glimpses the model executes on each image
        :param pixel_scaling: Defines how far the center of the glimpse can reach the borders
        :param mnist_size: Size of each image: MNIST_SIZE x MNIST_SIZE
        :param channels: Number of image channels
        :param scaling_factor: Scaling Factor of zooms
        :param sensorResolution: Resolution of the Sensor
        :param zooms: Resolution of the Sensor
        :param lr: The learning rate at epoch e=0
        :param lr_decay: Number of epochs after which the learning rate has linearly
                        decayed to min_lr
        :param min_lr: minimal learning rate
        :param momentum: should momentum be used
        :param loc_std: standard deviation of location policy
        """

        self.channels = channels # grayscale
        self.scaling = scaling_factor # zooms -> scaling * 2**<depth_level>
        self.sensorBandwidth = sensorResolution # fixed resolution of sensor
        self.depth = zooms # zooms
        self.output_dim = 10 # number of classes
        self.totalSensorBandwidth = totalSensorBandwidth
        self.batch_size = batch_size
        self.glimpses = glimpses

        # Learning Rate
        self.max_lr = lr
        self.min_lr = min_lr
        self.lr_decay_rate = lr_decay
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_type = lr_decay_type
        self.lr = lr
        self.step = 0

        # Learning
        self.optimizer = optimizer
        self.momentum = momentum

        # Location Policy Network
        self.pixel_scaling = pixel_scaling
        self.mnist_size = mnist_size

        self.eval_location_list = []

        # Size of Hidden state
        self.hs_size = 256

        # Learning Rate Decay
        if lr_decay_steps != 0 and self.lr_decay_type == "linear":
            self.lr_decay_rate = ((lr - min_lr) /
                                  lr_decay_steps)

        # Create Model
        outputs = self.model()

    def model(self):
        """
        Core Network of the RAM
        :return: Sequence of hidden states of the RNN
        """
        self.location_list = []
        self.location_mean_list = []
        self.location_stddev_list = []
        self.glimpses_list = []

        decoder = Decoder(self.hs_size, self.batch_size, self.pixel_scaling)
        decoder.initialize_hidden_state()

        # for i in self.glimpses:

        #outputs, _ = seq2seq.rnn_decoder(inputs, initial_state, lstm_cell, loop_function=self.get_next_input)

        #return outputs
        return 'Test'
