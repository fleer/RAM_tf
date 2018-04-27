import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.legacy_seq2seq as seq2seq

class RAM():
    """
    Neural Network class, that uses Tensorflow to build and train the Recurrent Attention Model
    """


    def __init__(self, totalSensorBandwidth, batch_size, optimizer, momentum, glimpses, pixel_scaling, mnist_size,
                 channels, scaling_factor, sensorResolution, zooms, lr, lr_decay, lr_decay_steps, lr_decay_type,
                 min_lr, session):
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

        self.session = session
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
        self.location_list = []
        self.location_mean_list = []
        self.location_stddev_list = []
        self.glimpses_list = []

        # Size of Hidden state
        self.hs_size = 256

        # Learning Rate Decay
        if lr_decay_steps != 0 and self.lr_decay_type == "linear":
            self.lr_decay_rate = ((lr - min_lr) /
                                  lr_decay_steps)

        # Tensorflow Placeholder
        self.inputs_placeholder = tf.placeholder(tf.float32, shape=([self.batch_size, self.mnist_size, self.mnist_size, 1]), name="images")
        self.actions = tf.placeholder(shape=[self.batch_size], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, 10, dtype=tf.float32)

        # If training --> False, only mean of location policy is used
        self.training = tf.placeholder(tf.bool, shape=[])

        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.summary_input = tf.summary.image("State", self.inputs_placeholder, max_outputs=1)

        # Global weights
        self.h_l_out = self.weight_variable((self.hs_size, 2))
        self.h_l_std_out = self.weight_variable((self.hs_size, 2))
        self.b_l_out = self.weight_variable((self.hs_size, 1))

        # Create Model
        outputs = self.model()

        # Get Model output & train
        self.cost_a, self.cost_l, self.cost_s, self.cost_b, self.reward, self.predicted_probs, self.train_a, self.train_b = self.loss(outputs)

    def get_images(self, X):
        """
        Get the glimpses created by the location policy network
        :param X: Input images
        :return: Tensorboard summary of the images and the corresponding glimpses
        """
        img = np.reshape(X, (self.batch_size, self.mnist_size, self.mnist_size, self.channels))
        feed_dict = {self.inputs_placeholder: img, self.training: False}#,
        fetches = [self.summary_input, self.summary_zooms]
        summary_image, summary_zooms = self.session.run(fetches, feed_dict=feed_dict)
        return summary_image, summary_zooms

    def evaluate(self,X,Y):
        """
        Evaluate the performance of the network
        :param X: Batch of images
        :param Y: Batch of the corresponding labels
        :return: Mean reward, predicted labels
        """
        feed_dict = {self.inputs_placeholder: X, self.actions: Y, self.training: True}
        fetches = [self.reward, self.predicted_probs]
        reward_fetched, predicted_labels_fetched = self.session.run(fetches, feed_dict=feed_dict)
        return reward_fetched, predicted_labels_fetched

    def train(self,X,Y):
        """
        Train the network
        :param X: Batch of images
        :param Y: Batch of the corresponding labels
        :return: Mean reward, predicted labels, accumulated loss, location policy loss, baseline loss
        """
        feed_dict = {self.inputs_placeholder: X, self.actions: Y, self.training: True, self.learning_rate: self.lr}
        fetches = [self.cost_a, self.cost_l, self.cost_s, self.cost_b, self.reward, self.predicted_probs, self.train_a, self.train_b]
        results = self.session.run(fetches, feed_dict=feed_dict)
        cost_a_fetched, cost_l_fetched, cost_s_fetched, cost_b_fetched, reward_fetched, prediction_labels_fetched, \
        train_a_fetched, train_b_fetched = results
        return reward_fetched, prediction_labels_fetched, cost_a_fetched, cost_l_fetched, cost_s_fetched, cost_b_fetched

    def get_next_input(self, output, i):
        """
        Loop function of recurrent attention network
        :param output: hidden state of recurrent network
        :param i: counter
        :return: next glimpse
        """
        mean_loc = self.hard_tanh(tf.matmul(tf.stop_gradient(output), self.h_l_out))
        std_loc = tf.nn.sigmoid(tf.matmul(tf.stop_gradient(output), self.h_l_std_out))
        # Clip location between [-1,1] and adjust its scale
        sample_loc =self.hard_tanh(mean_loc + tf.cond(self.training, lambda: tf.random_normal(mean_loc.get_shape(), 0, std_loc), lambda: 0. ))
        loc = sample_loc * self.pixel_scaling

        self.location_mean_list.append(tf.reduce_sum(mean_loc,1))
        self.location_stddev_list.append(tf.reduce_sum(std_loc,1))
        self.location_list.append(tf.reduce_sum(sample_loc,1))
        self.eval_location_list.append(loc)

        return self.Glimpse_Net(loc)

    def model(self):
        """
        Core Network of the RAM
        :return: Sequence of hidden states of the RNN
        """
        self.location_list = []
        self.location_mean_list = []
        self.glimpses_list = []
        # Create LSTM Cell
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hs_size, activation=tf.nn.relu, state_is_tuple=True)
        initial_state = lstm_cell.zero_state(self.batch_size, tf.float32)

        # Initial location mean generated by initial hidden state of RNN
        initial_loc = self.hard_tanh(tf.matmul(initial_state[0], self.h_l_out))
        #initial_std = self.hard_sigmoid(tf.matmul(initial_state[0], self.h_l_std_out))
        initial_std = tf.nn.sigmoid(tf.matmul(initial_state[0], self.h_l_std_out))
        sample_loc = self.hard_tanh(initial_loc + tf.cond(self.training, lambda: tf.random_normal(initial_loc.get_shape(), 0, initial_std), lambda: 0.))
        loc = sample_loc *self.pixel_scaling

        self.location_mean_list.append(tf.reduce_sum(initial_loc,1))
        self.location_stddev_list.append(tf.reduce_sum(initial_std,1))
        self.location_list.append(tf.reduce_sum(sample_loc,1))
        self.eval_location_list.append(loc)

        # Compute initial glimpse
        initial_glimpse = self.Glimpse_Net(loc)

        inputs = [initial_glimpse]
        inputs.extend([0] * (self.glimpses - 1))
        outputs, _ = seq2seq.rnn_decoder(inputs, initial_state, lstm_cell, loop_function=self.get_next_input)

        return outputs


    def loss(self, outputs):
        """
        Get classification and compute losses
        :param outputs: Sequence of hidden states of the RNN
        :return: accumulated loss, location policy loss, baseline loss, mean reward, predicted labels,
         gradient of hybrid loss, gradient of baseline loss
        """
        # Use mean baseline of all glimpses
        b_pred = []
        for o in outputs:
            o = tf.reshape(o, (self.batch_size, self.hs_size))
            b_pred.append(tf.squeeze(tf.matmul(o, self.b_l_out)))
        b = tf.transpose(tf.stack(b_pred),perm=[1,0])
        b_ng = tf.stop_gradient(b)


        # Initialize weights of action network
        a_h_out = self.weight_variable((self.hs_size, 10))

        # look at ONLY THE END of the sequence to predict label
        action_out = tf.nn.log_softmax(tf.matmul(tf.reshape(outputs[-1], (self.batch_size, self.hs_size)), a_h_out))
        max_p_y = tf.argmax(action_out, axis=-1)
        correct_y = tf.cast(self.actions, tf.int64)

        # reward per example
        R_batch = tf.cast(tf.equal(max_p_y, correct_y), tf.float32)
        R = tf.reshape(R_batch, (self.batch_size, 1))
        R = tf.stop_gradient(R)
        R = tf.tile(R, [1,self.glimpses])

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

        loc = tf.transpose(tf.stack(self.location_list),perm=[1,0])
        mean_loc = tf.transpose(tf.stack(self.location_mean_list),perm=[1,0,])
        std_loc = tf.transpose(tf.stack(self.location_stddev_list),perm=[1,0,])
        #TODO: Remove the summation of 2D Location while appending to list and evaluate the characteristic elegibility indiviually for each dimension

        Reinforce = tf.reduce_mean((loc - mean_loc)/tf.stop_gradient(std_loc)**2 * (R-b_ng))
        Reinforce_std = tf.reduce_mean((((loc - tf.stop_gradient(mean_loc))**2)-std_loc**2)/(std_loc**3) * (R-b_ng))

        # balances the scale of the two gradient components
        ratio = 0.75

        # Action Loss
        J = tf.reduce_sum(action_out * self.actions_onehot,axis=1)

        # Hybrid Loss
        # Scale the learning rate for the REINFORCE part by tf.stop_gradient(std_loc)**2, as suggested in (Williams, 1992)
        #cost = - tf.reduce_mean(J + ratio * tf.reduce_mean(tf.stop_gradient(std_loc))**2* (Reinforce+Reinforce_std), axis=0)
        cost = - tf.reduce_mean(J + ratio * (Reinforce+Reinforce_std), axis=0)

        # Baseline is trained with MSE
        b_loss = tf.losses.mean_squared_error(R, b)

        # Choose Optimizer
        if self.optimizer == "rmsprop":
            trainer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer == "adam":
            trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer == "adadelta":
            trainer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer == 'sgd':
            trainer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum, use_nesterov=True)
        else:
            raise ValueError("unrecognized update: {}".format(self.optimizer))

        # TODO: Implement gradient clipping
        train_op_a = trainer.minimize(cost)
        train_op_b = trainer.minimize(b_loss, var_list=[self.b_l_out])

        # Get data for Tensorboard summary
        take_first_zoom = []
        for gl in range(self.glimpses):
            take_first_zoom.append(self.glimpses_list[gl][0])
        self.summary_zooms = tf.summary.image("Zooms", tf.reshape(take_first_zoom, (self.glimpses, self.sensorBandwidth, self.sensorBandwidth, 1)), max_outputs=self.glimpses)

        return cost, -Reinforce, -Reinforce_std, b_loss, reward, action_out, train_op_a, train_op_b

    def weight_variable(self,shape):
        """
        Trainable network weights are initialized with uniform
        value within the range [-0.01, 0.01]
        :param shape: Desired shape
        :return: Tensorflow variable
        """
        initial = tf.random_uniform(shape, minval=-0.1, maxval=0.1)
        return tf.Variable(initial)

    def Glimpse_Net(self, location):
        """
        Glimpse Network
        :param location:
        :return: Glimpse Network output
        """
        # Number of weights
        hg_size = hl_size = 128
        g_size = 256

        # Initialize weights
        glimpse_hg = self.weight_variable((self.totalSensorBandwidth, hg_size))
        l_hl = self.weight_variable((2, hl_size))
        hg_g = self.weight_variable((hg_size, g_size))
        hl_g = self.weight_variable((hl_size, g_size))

        # Get glimpses
        glimpses = self.glimpseSensor(location)
        # Append glimpses to list for tensorboard summary
        self.glimpses_list.append(glimpses[0])

        # Process glimpses
        glimpses = tf.reshape(glimpses, [self.batch_size, self.totalSensorBandwidth])
        hg = tf.nn.relu(tf.matmul(glimpses, glimpse_hg))

        # Process locations
        hl = tf.nn.relu(tf.matmul(location, l_hl))

        # Combine glimpses and locations
        g = tf.nn.relu(tf.matmul(hg, hg_g) + tf.matmul(hl, hl_g))

        return g


    def glimpseSensor(self, normLoc):
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
            one_img = self.inputs_placeholder[k,:,:,:]

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
                    zoom = tf.image.resize_images(zoom, (self.sensorBandwidth, self.sensorBandwidth))
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

    def learning_rate_decay(self):
        """
        Function to control the linear decay
        of the learning rate
        :return: New learning rate
        """
        if self.lr_decay_type == "linear":
            # Linear Learning Rate Decay
            self.lr = max(self.min_lr, self.lr - self.lr_decay_rate)
        elif self.lr_decay_type == "exponential":
            # Exponential Learning Rate Decay
            self.lr = max(self.min_lr, self.max_lr * (self.lr_decay_rate ** self.step/self.lr_decay_steps))
        elif self.lr_decay_type == "exponential_staircase":
            # Exponential Learning Rate Decay
            self.lr = max(self.min_lr, self.max_lr * (self.lr_decay_rate ** (self.step // self.lr_decay_steps)))
        else:
            print("Wrong type of learning rate: " + self.lr_decay_type)
            return 0
        self.step += 1

        return self.lr
