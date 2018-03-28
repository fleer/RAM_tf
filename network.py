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
                 min_lr, loc_std, session):
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
        self.depth = zooms# zooms

        self.output_dim = 10
        self.totalSensorBandwidth = totalSensorBandwidth
        self.batch_size = batch_size
        self.glimpses = glimpses
        self.min_lr = min_lr
        self.lr_decay_rate = lr_decay
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_type = lr_decay_type
        self.lr = lr
        self.loc_std = loc_std
        self.pixel_scaling = pixel_scaling
        self.mnist_size = mnist_size
        self.location_list = []
        self.zoom_list = []
        self.step = 0
        self.optimizer = optimizer
        self.momentum = momentum

        # Size of Hidden state
        self.hs_size = 256



        # Learning Rate Decay
        if lr_decay_steps != 0 and self.lr_decay_type == "linear":
            self.lr_decay_rate = ((lr - min_lr) /
                                  lr_decay_steps)

        self.inputs_placeholder = tf.placeholder(tf.float32, shape=([self.batch_size, self.mnist_size, self.mnist_size, 1]), name="images")
        self.training = tf.placeholder(tf.bool, shape=[])
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.summary_input = tf.summary.image("State", self.inputs_placeholder, max_outputs=1)

        self.h_l_out = self.weight_variable((self.hs_size, 2))
        self.b_l_out = self.weight_variable((self.hs_size, 1))


        self.actions = tf.placeholder(shape=[self.batch_size], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, 10, dtype=tf.float32)


        outputs = self.model()
        self.cost_a, self.cost_l, self.cost_b, self.reward, self.predicted_labels, self.train_a, self.train_b = self.loss(outputs)

    def get_images(self, X):
        img = np.reshape(X, (self.batch_size, self.mnist_size, self.mnist_size, self.channels))
        feed_dict = {self.inputs_placeholder: img, self.training: False}#,
        fetches = [self.summary_input, self.summary_zooms]
        summary_image, summary_zooms = self.session.run(fetches, feed_dict=feed_dict)
        return summary_image, summary_zooms

    def evaluate(self,X,Y):
        feed_dict = {self.inputs_placeholder: X, self.actions: Y, self.training: False}#,
        fetches = [self.reward, self.predicted_labels]
        reward_fetched, predicted_labels_fetched = self.session.run(fetches, feed_dict=feed_dict)
        return reward_fetched, predicted_labels_fetched

    def train(self,X,Y):
        feed_dict = {self.inputs_placeholder: X, self.actions: Y, self.training: True, self.learning_rate: self.lr}#,
                     #self.actions_onehot: Y}
        fetches = [self.cost_a, self.cost_l, self.cost_b, self.reward, self.predicted_labels, self.train_a, self.train_b]

        results = self.session.run(fetches, feed_dict=feed_dict)

        cost_a_fetched, cost_l_fetched, cost_b_fetched, reward_fetched, prediction_labels_fetched, \
        train_a_fetched, train_b_fetched = results

        return reward_fetched, prediction_labels_fetched, cost_a_fetched, cost_l_fetched, cost_b_fetched

    def get_next_input(self, output, i):

        self.mean_loc = tf.stop_gradient(self.hard_tanh(tf.matmul(output, self.h_l_out)))

        sample_loc =self.mean_loc + tf.cond(self.training, lambda: tf.random_normal(self.mean_loc.get_shape(), 0, self.loc_std), lambda: 0. )

        self.loc = self.hard_tanh(sample_loc) * self.pixel_scaling
        self.location_list.append(self.loc)
        return self.Glimpse_Net(self.loc)

    def model(self):
        self.location_list = []
        self.zoom_list = []
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hs_size, activation=tf.nn.relu, state_is_tuple=True)

        initial_state = lstm_cell.zero_state(self.batch_size, tf.float32)

        initial_loc = self.hard_tanh(tf.matmul(initial_state[0], self.h_l_out))
        sample_loc = initial_loc + tf.cond(self.training, lambda: tf.random_normal(initial_loc.get_shape(), 0, self.loc_std), lambda: 0.)

        loc = self.hard_tanh(sample_loc) *self.pixel_scaling
        self.location_list.append(loc)

        initial_glimpse = self.Glimpse_Net(loc)


        inputs = [initial_glimpse]
        inputs.extend([0] * (self.glimpses - 1))

        outputs, _ = seq2seq.rnn_decoder(inputs, initial_state, lstm_cell, loop_function=self.get_next_input)

        return outputs


    def loss(self, outputs):
        a_h_out = self.weight_variable((self.hs_size, 10))

        # look at ONLY THE END of the sequence
        action_out = tf.nn.log_softmax(tf.matmul(tf.reshape(outputs[-1], (self.batch_size, self.hs_size)), a_h_out))

        # Use mean baseline of all glimpses
        b_pred = []
        for o in outputs:
            o = tf.reshape(o, (self.batch_size, self.hs_size))
            b_pred.append(tf.sigmoid(tf.matmul(o, self.b_l_out)))
        baseline = tf.reduce_mean(b_pred, axis=0)
        max_p_y = tf.argmax(action_out, axis=-1)
        correct_y = tf.cast(self.actions, tf.int64)

        R_batch = tf.cast(tf.equal(max_p_y, correct_y), tf.float32)  # reward per example

        reward = tf.reduce_mean(R_batch)  # overall reward

        R = tf.reshape(R_batch, (self.batch_size, 1))
        b = tf.reshape(baseline, (self.batch_size, 1))
        b_ng = tf.stop_gradient(b)

        Reinforce = (tf.reduce_mean(self.location_list, axis=0) - self.mean_loc)/(self.loc_std*self.loc_std) * (tf.tile(R,[1,2])-tf.tile(b_ng, [1,2]))
        ratio = 1.

        J = tf.concat([action_out * self.actions_onehot, ratio*Reinforce], 1)

        J = tf.reduce_sum(J,axis=1)
        J = tf.reduce_mean(J,axis=0)
        cost = -J

        b_loss = tf.losses.mean_squared_error(R, b)

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

        train_op_a = trainer.minimize(cost)
        train_op_b = trainer.minimize(b_loss, var_list=[self.b_l_out])

        take_first_zoom = []
        for gl in range(self.glimpses):
            take_first_zoom.append(self.zoom_list[gl][0])
        self.summary_zooms = tf.summary.image("Zooms", tf.reshape(take_first_zoom, (self.glimpses, self.sensorBandwidth, self.sensorBandwidth, 1)), max_outputs=self.glimpses)

        return cost, -Reinforce, b_loss, reward, max_p_y, train_op_a, train_op_b

    def weight_variable(self,shape):
        initial = tf.random_uniform(shape, minval=-0.01, maxval=0.01)
        return tf.Variable(initial)

    def Glimpse_Net(self, location):
        hg_size = hl_size = 128
        g_size = 256

        zooms = self.glimpseSensor(location)
        self.zoom_list.append(zooms[0])
        zooms = tf.reshape(zooms, (self.batch_size, self.totalSensorBandwidth))
        input_layer = tf.reshape(zooms, [self.batch_size, self.totalSensorBandwidth])

        glimpse_hg = self.weight_variable((self.totalSensorBandwidth, hg_size))
        hg = tf.nn.relu(tf.matmul(input_layer, glimpse_hg))

        l_hl = self.weight_variable((2, hl_size))
        hl = tf.nn.relu(tf.matmul(location, l_hl))

        hg_g = self.weight_variable((hg_size, g_size))
        hl_g = self.weight_variable((hl_size, g_size))
        g = tf.nn.relu(tf.matmul(hg, hg_g) + tf.matmul(hl, hl_g))

        return g


    def glimpseSensor(self, normLoc):
        # Convert location [-1,1] into MNIST Coordinates:
        loc = tf.round(((normLoc + 1.) / 2.) * self.mnist_size)
        loc = tf.cast(loc, tf.int32)
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

        return zooms

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
        if self.lr_decay_type == "linear":
            # Linear Learning Rate Decay
            self.lr = max(self.min_lr, self.lr - self.lr_decay_rate)
        elif self.lr_decay_type == "exponential":
            # Exponential Learning Rate Decay
            self.lr = max(self.min_lr, self.lr * (self.lr_decay_rate ** self.step/self.lr_decay_steps))
        elif self.lr_decay_type == "exponential_staircase":
            # Exponential Learning Rate Decay
            self.lr = max(self.min_lr, self.lr * (self.lr_decay_rate ** (int(self.step) // int(self.lr_decay_steps))))
            print(int(self.step) // int(self.lr_decay_steps))

        self.step += 1

        return self.lr
