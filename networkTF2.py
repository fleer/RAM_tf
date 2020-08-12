import os

import numpy as np
import tensorflow as tf

class AttentionControl(tf.keras.layers.Layer):
    """
    Loop function of recurrent attention network
    :return: next glimpse
    """
    def __init__(self, units, batch_size, pixel_scaling, mnist_size,
            sensorBandwidth, totalSensorBandwidth, depth):
        super(AttentionControl, self).__init__()
        # Number of weights
        hg_size = hl_size = 128
        g_size = 256
        self.pixel_scaling = pixel_scaling
        self.depth = depth
        self.batch_size = batch_size 
        self.eval_location_list = []
        self.location_list = []
        self.location_mean_list = []
        self.location_stddev_list = []
        self.glimpses_list = []
        self.baseline_list = []
        self.totalSensorBandwidth = totalSensorBandwidth
        self.sensorBandwidth = sensorBandwidth # fixed resolution of sensor
        self.training = 1
        self.mnist_size = mnist_size

        # Initialize weights
        self.h_location_std_out = tf.keras.layers.Dense(2, 
                kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.1))
        self.h_location_out = tf.keras.layers.Dense(2,
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
        self.first_glimpse = True

    def get_lists(self):
        return self.eval_location_list, self.location_list, self.location_mean_list, self.location_stddev_list, self.glimpses_list

    def call(self, inputs, output):
        #TODO: First location has to be random
        if (self.first_glimpse):
            mean_loc = tf.random.uniform(shape=(self.batch_size, 2), minval=-1, maxval=1)
            std_loc = tf.random.uniform(shape=(self.batch_size, 2), minval=-1, maxval=1)
            self.first_glimpse = False
        else:
            mean_loc = self.hard_tanh( self.h_location_out(output))
            std_loc = tf.nn.sigmoid( self.h_location_std_out(output))
            #mean_loc = self.hard_tanh( self.h_location_out(tf.stop_gradient(output)))
            #std_loc = tf.nn.sigmoid( self.h_location_std_out(tf.stop_gradient(output)))
        # Clip location between [-1,1] and adjust its scale
        #sample_loc = self.hard_tanh(mean_loc + tf.cond(self.training,
        #    lambda: tf.random.normal(mean_loc.get_shape(), 0, std_loc), lambda: 0. ))
        sample_loc = self.hard_tanh(mean_loc + tf.random.normal(mean_loc.get_shape(), 0, std_loc))
        sample_loc = tf.where(tf.math.is_nan(sample_loc), tf.zeros_like(sample_loc), sample_loc)
        loc = sample_loc * self.pixel_scaling
        glimpse = self.Glimpse_Net(loc, inputs)

        # Append stuff to lists
        self.location_mean_list.append(tf.reduce_sum(mean_loc,1))
        self.location_stddev_list.append(tf.reduce_sum(std_loc,1))
        self.location_list.append(tf.reduce_sum(sample_loc,1))
        self.glimpses_list.append(glimpse)
        self.eval_location_list.append(loc)

        return glimpse

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

        # # Process locations
        hl = self.h_location_layer(location)

        # # Combine glimpses and locations
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
            offset = self.sensorBandwidth* (self.pixel_scaling ** (self.depth-1))

            offset = tf.cast(offset, tf.int32)
            # pad image with zeros
            one_img = tf.image.pad_to_bounding_box(one_img, offset, offset, \
                                                   2*offset + self.mnist_size, 2*offset + self.mnist_size)

            # compute the different depth images
            for i in range(self.depth):

                # width/height of the next glimpse
                d = tf.cast(self.sensorBandwidth * (self.pixel_scaling ** i), tf.int32)
                r = d//2

                # get mean location
                loc_k = loc[k,:]
                adjusted_loc = offset + loc_k - r

                one_img2 = tf.reshape(one_img, (one_img.get_shape()[0], \
                                                 one_img.get_shape()[1]))

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

    def hard_tanh(self, x):
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

    def hard_sigmoid(self, x):
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

class Baseline(tf.keras.Model): 
    def __init__(self, units, batch_size):
        super(Baseline, self).__init__()
        self.batch_size = batch_size
        self.units = units

        # baseline
        self.baseline_layer = tf.keras.layers.Dense(1,
                kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.1))

    def call(self, outputs):

        # Use mean baseline of all glimpses
        b_pred = []
        for o in outputs:
            o = tf.reshape(tf.stop_gradient(o), (self.batch_size, self.units))
            b_pred.append(tf.squeeze(self.baseline_layer(o)))
            b = tf.transpose(tf.stack(b_pred),perm=[1,0])
        return b

class RAM(tf.keras.Model):
    def __init__(self, units, batch_size, pixel_scaling, mnist_size,
            sensorBandwidth, totalSensorBandwidth, depth, num_glimpses):
        super(RAM, self).__init__()
        self.batch_size = batch_size
        self.units = units
        self.glimpses = num_glimpses

        self.lstm = tf.keras.layers.LSTMCell(units, activation=tf.nn.relu, recurrent_initializer='zeros')

        # classification
        self.classification_layer = tf.keras.layers.Dense(10,
                activation=tf.nn.log_softmax,
                kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.1))
        # baseline
        self.baseline_layer = tf.keras.layers.Dense(1,
                kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.1))
        # used for attention
        self.attention = AttentionControl(units, batch_size,
                pixel_scaling, mnist_size, sensorBandwidth, totalSensorBandwidth, depth)

    def reset_attention(self):
        self.attention.reset_lists()

    def get_attention_lists(self):
        return self.attention.get_lists()

    def call(self, inputs):
        outputs = []
        output = tf.zeros((self.batch_size, self.units))
        hidden = [output, output]
        #self.lstm.reset_states(states=[output, output])
        self.attention.reset_lists()
        for g in range(self.glimpses):
            glimpse = self.attention(inputs, output)
            output, hidden = self.lstm(glimpse, hidden)
            outputs.append(output)
        # look at ONLY THE END of the sequence to predict label
        action_out = self.classification_layer(output)

        return glimpse, action_out, outputs

    def loss(self, correct_y, action_out, baseline):
        """
        Get classification and compute losses
        :param outputs: Sequence of hidden states of the RNN
        :return: accumulated loss, location policy loss, baseline loss, mean reward, predicted labels,
         gradient of hybrid loss, gradient of baseline loss
        """

        max_p_y = tf.argmax(action_out, axis=-1)
        actions_onehot = tf.one_hot(max_p_y, 10, dtype=tf.float32)
        # reward per example
        R_batch = tf.cast(tf.equal(max_p_y, correct_y), tf.float32)
        R = tf.reshape(R_batch, (self.batch_size, 1))
        R = tf.stop_gradient(R)
        R = tf.tile(R, [1, self.glimpses])

        # mean reward
        reward = tf.reduce_mean(R_batch)

        # REINFORCE algorithm for policy network loss
        # -------
        # Williams, Ronald J. "Simple statistical gradient-following
        # algorithms for connectionist reinforcement learning."
        # Machine learning 8.3-4 (1992): 229-256.
        # -------
        # characteristic eligibility taken from sec 6. p.237-239
        #
        # d ln(f(m,s,x))   (x - m)
        # -------------- = -------- with m = mean, x = sample, s = standard deviation
        #       d m          s**2
        #

        loc = tf.transpose(tf.stack(self.attention.location_list),perm=[1,0])
        mean_loc = tf.transpose(tf.stack(self.attention.location_mean_list),perm=[1,0,])
        std_loc = tf.transpose(tf.stack(self.attention.location_stddev_list),perm=[1,0,])
        #TODO: Remove the summation of 2D Location while appending to list and evaluate the characteristic elegibility indiviually for each dimension

        Reinforce = tf.reduce_mean((loc -
            mean_loc)/tf.stop_gradient(std_loc)**2 * (R - baseline))
        Reinforce_std = tf.reduce_mean((((loc -
            tf.stop_gradient(mean_loc))**2)-std_loc**2)/(std_loc**3) *
            (R-baseline))

        # balances the scale of the two gradient components
        ratio = 0.75

        # Action Loss
        J = tf.reduce_sum(action_out * actions_onehot,axis=1)

        # Hybrid Loss
        # Scale the learning rate for the REINFORCE part by tf.stop_gradient(std_loc)**2, as suggested in (Williams, 1992)
        #cost = - tf.reduce_mean(J + ratio * tf.reduce_mean(tf.stop_gradient(std_loc))**2* (Reinforce+Reinforce_std), axis=0)
        cost = - tf.reduce_mean(J + ratio * (Reinforce+Reinforce_std), axis=0)

        b_loss = tf.keras.losses.MSE(R, baseline)

        return cost, -Reinforce, -Reinforce_std, b_loss, R
