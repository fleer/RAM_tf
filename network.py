import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib import seq2seq
import tensorflow.contrib.legacy_seq2seq as seq2seq

class RAM():
    """
    Neural Network class, that uses KERAS to build and trains the Recurrent Attention Model
    """


    def __init__(self, totalSensorBandwidth, batch_size, glimpses, pixel_scaling, mnist_size, channels, scaling_factor, sensorResolution, zooms, lr, lr_decay, min_lr, loc_std, session):
        """
        Intialize parameters, determine the learning rate decay and build the RAM
        :param totalSensorBandwidth: The length of the networks input vector
                                    ---> nZooms * sensorResolution * sensorResolution * channels
        :param batch_size: Size of each batch
        :param glimpses: Number of glimpses the model executes on each image
        :param optimizer: The used optimize: "sgd, rmsprop, adadelta, adam,..."
        :param lr: The learning rate at epoch e=0
        :param lr_decay: Number of epochs after which the learning rate has linearly
                        decayed to min_lr
        :param min_lr: minimal learning rate
        :param momentum: should momentum be used
        :param loc_std: standard deviation of location policy
        :param clipnorm: Gradient clipping
        :param clipvalue: Gradient clipping
        """

        self.session = session
        self.channels = channels # grayscale
        self.scaling = scaling_factor # zooms -> scaling * 2**<depth_level>
        self.sensorBandwidth = sensorResolution # fixed resolution of sensor
        self.depth = zooms# zooms

        self.output_dim = 10
        self.totalSensorBandwidth = totalSensorBandwidth
        self.batch_size = batch_size
        self.glimpses = glimpses
        self.min_lr = min_lr
        self.lr_decay = lr_decay
        self.lr = lr
        self.loc_std = loc_std
        self.pixel_scaling = pixel_scaling
        self.mnist_size = mnist_size

        # Learning Rate Decay
        if self.lr_decay != 0:
            self.lr_decay_rate = ((lr - min_lr) /
                                  lr_decay)

        self.inputs_placeholder = tf.placeholder(tf.float32, shape=([self.batch_size, self.mnist_size, self.mnist_size, 1]), name="images")
        #self.inputs_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size, self.totalSensorBandwidth), name="images")

        self.h_l_out = self.weight_variable((256, 2))
        self.b_l_out = self.weight_variable((256, 1))


        self.actions = tf.placeholder(shape=[self.batch_size], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, 10, dtype=tf.float32)


        outputs = self.model()
        #outputs = self.Glimpse_Net(tf.zeros([self.batch_size, 2]))
        self.cost_a, self.cost_l, self.cost_b, self.reward, self.predicted_labels, self.train_a, self.train_l, self.train_b = self.calc_reward(outputs)

    def evaluate(self,X,Y):
        feed_dict = {self.inputs_placeholder: X, self.actions: Y}#,
        fetches = [self.reward, self.predicted_labels]
        reward_fetched, predicted_labels_fetched = self.session.run(fetches, feed_dict=feed_dict)
        return reward_fetched, predicted_labels_fetched

    def train(self,X,Y):

        feed_dict = {self.inputs_placeholder: X, self.actions: Y}#,
                     #self.actions_onehot: Y}
        fetches = [self.cost_a, self.cost_l, self.cost_b, self.reward, self.predicted_labels, self.train_a, self.train_l, self.train_b]

        results = self.session.run(fetches, feed_dict=feed_dict)

        cost_a_fetched, cost_l_fetched, cost_b_fetched, reward_fetched, prediction_labels_fetched, \
        train_a_fetched, train_l_fetched, train_b_fetched= results
        return reward_fetched, prediction_labels_fetched, cost_a_fetched, cost_l_fetched, cost_b_fetched



    def calc_reward(self, outputs):
        outputs = outputs[-1]  # look at ONLY THE END of the sequence
        outputs = tf.reshape(outputs, (self.batch_size, 256))

        a_h_out = self.weight_variable((256, 10))
        action_out = tf.matmul(outputs, a_h_out)

       # action_out = tf.layers.Dense(units=10,
       #                                       #activation=tf.nn.softmax,#tf.nn.log_softmax,
       #                                       kernel_initializer=tf.initializers.random_uniform(-0.1, 0.1),
       #                                       bias_initializer=tf.initializers.random_uniform(-0.1, 0.1),
       #                                       )(outputs)


        #baseline = tf.nn.sigmoid(tf.matmul(outputs, self.b_l_out))
        baseline = tf.matmul(outputs, self.b_l_out)

        max_p_y = tf.argmax(action_out, axis=-1)
        correct_y = tf.cast(self.actions, tf.int64)

        R_batch = tf.cast(tf.equal(max_p_y, correct_y), tf.float32)  # reward per example

        reward = tf.reduce_mean(R_batch)  # overall reward

        R = tf.reshape(R_batch, (self.batch_size, 1))
        b = tf.reshape(baseline, (self.batch_size, 1))

        Reinforce = (self.loc - self.mean_loc)/(self.loc_std*self.loc_std) * (tf.tile(R,[1,2])-tf.tile(b, [1,2]))

        J = tf.losses.softmax_cross_entropy(self.actions_onehot,action_out)
        #J =  self.actions_onehot * action_out
        #J = tf.reduce_sum(J, axis=-1)
        #J = tf.reduce_mean(J, axis=0)
        #cost = -J

        b_loss = tf.losses.mean_squared_error(R, b)
        optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)
        train_op_a = optimizer.minimize(J)
        train_op_l = optimizer.minimize(-tf.reduce_mean(Reinforce), var_list=[self.h_l_out])
        train_op_b = optimizer.minimize(b_loss, var_list=[self.b_l_out])


        return -J, -Reinforce, b_loss, reward, max_p_y, train_op_a, train_op_l, train_op_b

    def weight_variable(self,shape):
        initial = tf.random_uniform(shape, minval=-0.1, maxval=0.1)
        return tf.Variable(initial)

    def Glimpse_Net(self, location):
        hg_size = hl_size = 128
        g_size = 256

        zooms = self.glimpseSensor(location)
        input_layer = tf.reshape(zooms, [self.batch_size, self.totalSensorBandwidth])

        glimpse_hg = self.weight_variable((self.totalSensorBandwidth, hg_size))
        hg = tf.nn.relu(tf.matmul(input_layer, glimpse_hg))

        l_hl = self.weight_variable((2, hl_size))
        hl = tf.nn.relu(tf.matmul(location, l_hl))
        hg_g = self.weight_variable((hg_size, g_size))
        hl_g = self.weight_variable((hl_size, g_size))

        g = tf.nn.relu(tf.matmul(hg, hg_g) + tf.matmul(hl, hl_g))

      #  hg_1 = self.weight_variable((hg_size + hl_size, g_size))
      #  hg_2 = self.weight_variable((g_size, g_size))
      #  concat = tf.concat([hg,hl], axis=-1)
      #  g_1 = tf.nn.relu(tf.matmul(concat, hg_1))
      #  g = tf.matmul(g_1, hg_2)


        return g

    def get_next_input(self, output, i):

        self.mean_loc = self.hard_tanh(tf.matmul(output, self.h_l_out))


        self.loc = self.mean_loc + tf.random_normal(self.mean_loc.get_shape(), 0, self.loc_std)

        sample_loc = self.hard_tanh(self.loc) * self.pixel_scaling
        return self.Glimpse_Net(sample_loc)

    def model(self):
        initial_loc = self.hard_tanh(tf.matmul(tf.zeros((self.batch_size,256)), self.h_l_out))

        sample_loc = initial_loc + tf.random_normal(initial_loc.get_shape(), 0, self.loc_std)

        initial_loc = self.hard_tanh(sample_loc) *self.pixel_scaling

        initial_glimpse = self.Glimpse_Net(initial_loc)

        lstm_cell = tf.nn.rnn_cell.BasicRNNCell(256, activation=tf.nn.relu)

        initial_state = lstm_cell.zero_state(self.batch_size, tf.float32)

        inputs = [initial_glimpse]
        inputs.extend([0] * (self.glimpses - 1))

        outputs, _ = seq2seq.rnn_decoder(inputs, initial_state, lstm_cell, loop_function=self.get_next_input)
        #self.get_next_input(outputs[-1], 0)

        return outputs

    def glimpseSensor(self, normLoc):
        # assert not np.any(np.isnan(normLoc))," Locations have to be between 1, -1: {}".format(normLoc)
        # assert np.any(np.abs(normLoc)<=1)," Locations have to be between 1, -1: {}".format(normLoc)


        # Convert location [-1,1] into MNIST Coordinates:
        loc = tf.round(((normLoc + 1.) / 2.) * self.mnist_size)
        loc = tf.cast(loc, tf.int32)

        #img = tf.reshape(self.inputs_placeholder, (self.batch_size, self.mnist_size, self.mnist_size, self.channels))

        zooms = []

        # process each image individually
        for k in range(self.batch_size):
            imgZooms = []
            one_img = self.inputs_placeholder[k,:,:,:]
            #one_img = img[k,:,:,:]
            offset = self.sensorBandwidth* (self.scaling ** (self.depth-1))

            # pad image with zeros
            one_img = tf.image.pad_to_bounding_box(one_img, offset, offset, \
                                                   2*offset + self.mnist_size, 2*offset + self.mnist_size)

            for i in range(self.depth):
                d = tf.cast(self.sensorBandwidth * (self.scaling ** i), tf.int32)
                r = d//2

                loc_k = loc[k,:]
                adjusted_loc = offset + loc_k - r

                one_img2 = tf.reshape(one_img, (one_img.get_shape()[0].value, \
                                                one_img.get_shape()[1].value))

                # crop image to (d x d)
                zoom = tf.slice(one_img2, adjusted_loc, [d,d])
                #zoom = one_img2[adjusted_loc[0]:adjusted_loc[0]+d, adjusted_loc[1]:adjusted_loc[1]+d]
                #assert not K.any(np.equal(zoom.shape, (0,0))), "Picture has size 0, location {}, depth {}".format(adjusted_loc, d)
                #assert len(zoom[0]) == d and len(zoom[1]) == d, "Glimpse has the dims: {}".format(zoom.shape)

                # resize cropped image to (sensorBandwidth x sensorBandwidth)
                if i > 0:
                    #zoom = cv2.resize(zoom, (self.sensorBandwidth, self.sensorBandwidth),
                    #                  interpolation=cv2.INTER_LINEAR)
                    #zoom = tf.cast(zoom, tf.int32)
                    zoom = tf.reshape(zoom, (1, d, d, 1))
                    zoom = tf.image.resize_images(zoom, (self.sensorBandwidth, self.sensorBandwidth))
                    zoom = tf.reshape(zoom, (self.sensorBandwidth, self.sensorBandwidth))
                imgZooms.append(zoom)
            zooms.append(tf.stack(imgZooms))

        #  shapes = set(arr.shape for arr in zooms)
        #  assert len(shapes) == 1, "zooms have different shapes: {}".format(zooms)
        zooms = tf.stack(zooms)

        return tf.reshape(zooms, (self.batch_size, self.totalSensorBandwidth))

    def hard_tanh(self, x):
        """Segment-wise linear approximation of tanh.

         Faster than tanh.
         Returns `-1.` if `x < -1.`, `1.` if `x > 1`.
         In `-1. <= x <= 1.`, returns `x`.

         # Arguments
             x: A tensor or variable.

         # Returns
             A tensor.
         """

        lower = tf.convert_to_tensor(-1., x.dtype.base_dtype)
        upper = tf.convert_to_tensor(1., x.dtype.base_dtype)
        x = tf.clip_by_value(x, lower, upper)
        return x

    def learning_rate_decay(self):
        """
        Function to control the linear decay
        of the learning rate
        :return: New learning rate
        """
        # Linear Learning Rate Decay
        self.lr = max(self.min_lr, self.lr - self.lr_decay_rate)
        return self.lr
