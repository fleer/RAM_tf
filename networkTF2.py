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

class attentionControl(tf.keras.layers.Layer):
    """
    Loop function of recurrent attention network
    :param output: hidden state of recurrent network
    :param i: counter
    :return: next glimpse
    """
    def __init__(self, units, pixel_scaling):
        super(attentionControl, self).__init__()
        self.pixel_scaling = pixel_scaling
        self.eval_location_list = []
        self.location_list = []
        self.location_mean_list = []
        self.location_stddev_list = []
        self.glimpses_list = []
        
        self.b_l_out = tf.keras.layers.Dense(units, kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.1))
        self.h_l_out = tf.keras.layers.Dense(units, kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.1))

        # Initialize weights
        tf.keras.layers.Dense(32, input_shape=(batch_size,), 
        kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.1)) 
        glimpse_hg = self.weight_variable((self.totalSensorBandwidth, hg_size))
        l_hl = self.weight_variable((2, hl_size))
        hg_g = self.weight_variable((hg_size, g_size))
        hl_g = self.weight_variable((hl_size, g_size))

        self.h_l_std_out = tf.keras.layers.Dense(units, kernel_initializer=tf.kernel_initializer(mean=0.1))


    def call(self, output):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        mean_loc = hard_tanh( self.h_l_out(tf.stop_gradient(output)))
        std_loc = tf.nn.sigmoid( self.h_l_std_out(tf.stop_gradient(output)))
        # Clip location between [-1,1] and adjust its scale
        sample_loc = hard_tanh(mean_loc + tf.cond(self.training, lambda: tf.random.normal(mean_loc.get_shape(), 0, std_loc), lambda: 0. ))
        loc = sample_loc * self.pixel_scaling
        baseline = self.b_l_out(output);

        self.location_mean_list.append(tf.reduce_sum(mean_loc,1))
        self.location_stddev_list.append(tf.reduce_sum(std_loc,1))
        self.location_list.append(tf.reduce_sum(sample_loc,1))
        self.eval_location_list.append(loc)

        return output, self.Glimpse_Net(loc), baseline
    
    def Glimpse_Net(self, location):
        """
        Glimpse Network
        :param location:
        :return: Glimpse Network output
        """
        # Number of weights
        hg_size = hl_size = 128
        g_size = 256

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

class Decoder(tf.keras.Model):
    def __init__(self, dec_units, batch_sz, pixel_scaling):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.lstm = tf.keras.layers.LSTM(self.dec_units, activation=tf.nn.relu,
                return_sequences=True, return_state=True,
                recurrent_initializer='zeros')

        # used for attention
        self.attention = attentionControl(self.dec_units, pixel_scaling)
        print('Decoder Initialized')
    
    def getInitialState(self):
        return self.lstm.get_initial_state()

    def call(self, x, hidden, enc_output):
     #   # enc_output shape == (batch_size, max_length, hidden_size)
     #   context_vector, attention_weights = self.attention(hidden, enc_output)

     #   # x shape after passing through embedding == (batch_size, 1, embedding_dim)
     #   x = self.embedding(x)

     #   # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
     #   x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

     #   # passing the concatenated vector to the GRU
     #   output, state = self.lstm(x)

     #   # output shape == (batch_size * 1, hidden_size)
     #   output = tf.reshape(output, (-1, output.shape[2]))

     #   # output shape == (batch_size, vocab)
     #   x = self.fc(output)

     #   return x, state, attention_weights
     pass

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
        # Initial location mean generated by initial hidden state of RNN
        #initial_loc = hard_tanh(tf.matmul(initial_state[0], self.h_l_out))
        #initial_std = self.hard_sigmoid(tf.matmul(initial_state[0], self.h_l_std_out))
        #initial_std = tf.nn.sigmoid(tf.matmul(initial_state[0], self.h_l_std_out))
        #sample_loc = self.hard_tanh(initial_loc + tf.cond(self.training, lambda: tf.random_normal(initial_loc.get_shape(), 0, initial_std), lambda: 0.))
        #loc = sample_loc * self.pixel_scaling

        #self.location_mean_list.append(tf.reduce_sum(initial_loc,1))
        #self.location_stddev_list.append(tf.reduce_sum(initial_std,1))
        #self.location_list.append(tf.reduce_sum(sample_loc,1))
        #self.eval_location_list.append(loc)

        # Compute initial glimpse
        #initial_glimpse = self.Glimpse_Net(loc)

        #inputs = [initial_glimpse]
        #inputs.extend([0] * (self.glimpses - 1))
        #outputs, _ = seq2seq.rnn_decoder(inputs, initial_state, lstm_cell, loop_function=self.get_next_input)

        #return outputs
        return 'Test'