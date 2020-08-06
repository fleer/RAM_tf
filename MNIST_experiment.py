from MNIST_Processing import MNIST
from networkTF2 import RAM
import numpy as np
import tensorflow as tf
from collections import defaultdict
import logging
import time
import os
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pdb

class Experiment():
    """
    Main class, controlling the experiment
    """

    results = defaultdict(list)


    def __init__(self, PARAMETERS, DOMAIN_OPTIONS):

        logging.basicConfig(level=logging.INFO)

        #   ================
        #   Reading the parameters
        #   ================

        self.batch_size = PARAMETERS.BATCH_SIZE
        self.max_epochs = PARAMETERS.MAX_EPOCHS
        self.M = DOMAIN_OPTIONS.MONTE_CARLO
        self.test_images = []

        # Compute the ratio converting unit width in the coordinate system to the number of pixels.
        # -----------------------------------
        # Ba, J. L., Mnih, V., Deepmind, G., & Kavukcuoglu, K. (n.d.).
        # MULTIPLE OBJECT RECOGNITION WITH VISUAL ATTENTION.
        # Retrieved from https://arxiv.org/pdf/1412.7755.pdf
        # -----------------------------------
        # This ratio presents an exploration versus exploitation trade off.
        if DOMAIN_OPTIONS.TRANSLATE:
            pixel_scaling = (DOMAIN_OPTIONS.UNIT_PIXELS * 2.)/ float(DOMAIN_OPTIONS.TRANSLATED_MNIST_SIZE)
        else:
            pixel_scaling = (DOMAIN_OPTIONS.UNIT_PIXELS * 2.)/ float(DOMAIN_OPTIONS.MNIST_SIZE)

        # Standard or Translated MNIST-Dataset
        if DOMAIN_OPTIONS.TRANSLATE:
            mnist_size = DOMAIN_OPTIONS.TRANSLATED_MNIST_SIZE
        else:
            mnist_size = DOMAIN_OPTIONS.MNIST_SIZE

        totalSensorBandwidth = DOMAIN_OPTIONS.DEPTH * DOMAIN_OPTIONS.SENSOR * DOMAIN_OPTIONS.SENSOR * DOMAIN_OPTIONS.CHANNELS

        #   ================
        #   Loading the MNIST Dataset
        #   ================

        self.mnist = MNIST(DOMAIN_OPTIONS.MNIST_SIZE, self.batch_size, DOMAIN_OPTIONS.TRANSLATE, DOMAIN_OPTIONS.TRANSLATED_MNIST_SIZE, DOMAIN_OPTIONS.MONTE_CARLO)

        # Create file writer
        summary_writer = tf.summary.create_file_writer("summary")

        #   ================
        #   Creating the RAM
        #   ================
        #   self.ram = RAM(totalSensorBandwidth, self.batch_size*self.M, PARAMETERS.OPTIMIZER, PARAMETERS.MOMENTUM,
        #         DOMAIN_OPTIONS.NGLIMPSES, pixel_scaling, mnist_size, DOMAIN_OPTIONS.CHANNELS, DOMAIN_OPTIONS.SCALING_FACTOR,
        #         DOMAIN_OPTIONS.SENSOR, DOMAIN_OPTIONS.DEPTH,
        #         PARAMETERS.LEARNING_RATE, PARAMETERS.LEARNING_RATE_DECAY, PARAMETERS.LEARNING_RATE_DECAY_STEPS, PARAMETERS.LEARNING_RATE_DECAY_TYPE,
        #         PARAMETERS.MIN_LEARNING_RATE)

        self.ram = Decoder(256, self.batch_size*self.M, pixel_scaling, mnist_size, DOMAIN_OPTIONS.SENSOR, totalSensorBandwidth, DOMAIN_OPTIONS.DEPTH)

        # TODO: Port to TF2
    #    self.saver = tf.train.Saver(max_to_keep=5)
    #    if PARAMETERS.LOAD_MODEL == True:
    #        print ('Loading Model...')
#   #            # ckpt = tf.train.get_checkpoint_state(PARAMETERS.MODEL_FILE_PATH)
    #        #self.saver.restore(sess, ckpt.model_checkpoint_path)
    #        self.saver.restore(sess, PARAMETERS.MODEL_FILE_PATH)
    #    else:
    #        sess.run(tf.global_variables_initializer())

    #    #   ================
    #    #   Train
    #    #   ================
        self.train(PARAMETERS.EARLY_STOPPING, PARAMETERS.PATIENCE)
    #    self.save('./', 'results.json')

    def performance_run(self, total_epochs, validation=False):
        """
        Function for evaluating the current model on the
        validation- or test-dataset

        :param total_epochs: Number of trained epochs
        :param validation: Should the smaller validation-dataset
                be evaluated
        :return: current accuracy and its error
        """
        actions = 0.
        actions_sqrt = 0.
        if validation:
            num_data = len(self.mnist.dataset.validation._images)
            batches_in_epoch = num_data // self.batch_size
        else:
            num_data = len(self.mnist.dataset.test._images)
            batches_in_epoch = num_data // self.batch_size

        for _ in range(batches_in_epoch):
            if validation:
                X, Y, Y_S = self.mnist.get_batch(self.batch_size, data_type="validation")
            else:
                X, Y, Y_S = self.mnist.get_batch(self.batch_size, data_type="test")
                self.test_images = X

            _, pred_action = self.ram.evaluate(X,Y)

            # Get Mean of the M samples for the same data for evaluating performance
            # -----------------------------------
            # Ba, J. L., Mnih, V., Deepmind, G., & Kavukcuoglu, K. (n.d.).
            # MULTIPLE OBJECT RECOGNITION WITH VISUAL ATTENTION.
            # Retrieved from https://arxiv.org/pdf/1412.7755.pdf
            # -----------------------------------
            # See Eq. (14)
            # As the the location prediction is stochastic, the attention model can be
            # evaluated multiple times on the same sample.
            # For evaluation, the mean of the log probabilities is then used for class prediction

            pred_action = np.reshape(pred_action,
                                     [self.M, -1, 10])
            pred_action = np.mean(pred_action, 0)
            pred_labels = np.argmax(pred_action, -1)
            actions += np.sum(np.equal(pred_labels,Y_S).astype(np.float32), axis=-1)
            actions_sqrt += np.sum((np.equal(pred_labels,Y_S).astype(np.float32))**2, axis=-1)

        accuracy = actions/(num_data)
        accuracy_std = np.sqrt(((actions_sqrt/(num_data)) - accuracy**2)/(num_data))

        if not validation:
            # Save to results file
            self.results['learning_steps'].append(total_epochs)
            self.results['accuracy'].append(accuracy)
            self.results['accuracy_std'].append(accuracy_std)

        return accuracy, accuracy_std

    def train(self, early_stopping, patience):
        """
        Training the current model
        :param early_stopping: Use early stopping
        :param patience: Number of Epochs observing the worsening of
                Validation set, before stopping
        :param session: Tensorflow session
        :return:
        """

        total_epochs = 0
        validation_accuracy = 0
        # Initial Performance Check
        # performance_accuracy, performance_accuracy_std = self.performance_run(total_epochs)
        # logging.info("Epoch={:d}: >>> Test-Accuracy: {:.4f} "
        #               "+/- {:.6f}".format(total_epochs, performance_accuracy, performance_accuracy_std))
        num_train_data = len(self.mnist.dataset.train._images)

        patience_steps = 0
        early_stopping_accuracy = 0.
        visualize_classification = True
        for _ in range(self.max_epochs):
            # summary = tf.Summary()
            start_time = time.time()
            train_accuracy = 0
            train_accuracy_sqrt = 0
            a_loss = []
            l_loss = []
            s_loss = []
            b_loss = []
            # while total_epochs == self.mnist.dataset.train.epochs_completed:
            X, Y, _= self.mnist.get_batch(self.batch_size, data_type="train")
        #        _, pred_action, nnl_loss, reinforce_loss, reinforce_std_loss, baseline_loss = self.ram.train(X,Y)
            if (visualize_classification):
                self.visualize(X[:self.batch_size],Y[:self.batch_size], Y[:self.batch_size]); visualize_classification = False
                hidden = self.ram.initialize_hidden_state()
                self.ram(X,hidden)
                # pred_action = np.argmax(pred_action, -1)
                # train_accuracy += np.sum(np.equal(pred_action,Y).astype(np.float32), axis=-1)
                # train_accuracy_sqrt+= np.sum((np.equal(pred_action,Y).astype(np.float32))**2, axis=-1)
                # a_loss.append(nnl_loss)
                # l_loss.append(reinforce_loss)
                # s_loss.append(reinforce_std_loss)
                # b_loss.append(baseline_loss)
            # total_epochs += 1
            # lr = self.ram.learning_rate_decay()

        #    # Train Accuracy
        #    train_accuracy = train_accuracy/(num_train_data*self.M)

        #    if total_epochs % 10 == 0:
        #        # Test Accuracy
        #        performance_accuracy, performance_accuracy_std = self.performance_run(total_epochs)

        #        # Print out Infos
        #        logging.info("Epoch={:d}: >>> Test-Accuracy: {:.4f} +/- {:.6f}".format(total_epochs, performance_accuracy, performance_accuracy_std))

        #        # Some visualization
        #        img, zooms = self.ram.get_images(np.vstack([self.test_images[0]]*self.batch_size*self.M))

        #        self.summary_writer.add_summary(img, total_epochs)
        #        self.summary_writer.add_summary(zooms, total_epochs)
        #        self.test_images = []
        #    else:
        #        # Validation Accuracy
        #        validation_accuracy, vaidation_accuracy_std = self.performance_run(total_epochs, validation=True)

        #        train_accuracy_std = np.sqrt(((train_accuracy_sqrt/(num_train_data*self.M)) - train_accuracy**2)/(num_train_data*self.M))

        #        # Print out Infos
        #        logging.info("Epoch={:d}: >>> examples/s: {:.2f}, Accumulated-Loss: {:.4f}, Location-Mean Loss: {:.4f}, Location-Stddev Loss: {:.4f}, Baseline-Loss: {:.4f}, "
        #                     "Learning Rate: {:.6f}, Train-Accuracy: {:.4f} +/- {:.6f}, "
        #                     "Validation-Accuracy: {:.4f} +/- {:.6f}".format(total_epochs,
        #                         float(num_train_data)/float(time.time()-start_time), np.mean(a_loss), np.mean(l_loss), np.mean(s_loss), np.mean(b_loss),
        #                         lr, train_accuracy, train_accuracy_std, validation_accuracy, vaidation_accuracy_std))

        #        # Early Stopping
        #        if early_stopping and early_stopping_accuracy < validation_accuracy:
        #            early_stopping_accuracy = validation_accuracy
        #            patience_steps = 0
        #        else:
        #            patience_steps += 1

        #    # Gather information for Tensorboard
        #    summary.value.add(tag='Losses/Accumulated Loss', simple_value=float(np.mean(a_loss)))
        #    summary.value.add(tag='Losses/Location: Mean Loss', simple_value=float(np.mean(l_loss)))
        #    summary.value.add(tag='Losses/Location: Stddev Loss', simple_value=float(np.mean(s_loss)))
        #    summary.value.add(tag='Losses/Baseline Loss', simple_value=float(np.mean(b_loss)))
        #    summary.value.add(tag='Accuracy/Performance', simple_value=float(performance_accuracy))
        #    summary.value.add(tag='Accuracy/Validation', simple_value=float(validation_accuracy))
        #    summary.value.add(tag='Accuracy/Train', simple_value=float(train_accuracy))

        #    self.summary_writer.add_summary(summary, total_epochs)

        #    self.summary_writer.flush()

        #    # Early Stopping
        #    if patience_steps > patience:
        #        self.saver.save(session, './Model/best_model-' + str(total_epochs) + '.cptk')
        #        logging.info("Early Stopping at Epoch={:d}! Validation Accuracy is not increasing. The best Newtork will be saved!".format(total_epochs))
        #        return 0

        #    # Save Model
        #    if total_epochs % 100 == 0:
        #        self.saver.save(session, save_path='./Model', global_step=total_epochs)

    def visualize(self, batch_x, batch_y, batch_pred):
        """Plot a dictionary of figures.

        Parameters
        ----------
        figures : <title, figure> dictionary
        ncols : number of columns of subplots wanted in the display
        nrows : number of rows of subplots wanted in the figure
        """
        plt.ion()
        n = int(np.sqrt(len(batch_x)))

        # TODO: Why?
        try:
            if not self._fig is None:
                # self._fig.clear()
                # self._fig, self.axeslist = plt.subplots(ncols=n+1, nrows=n+1)
                print("Test")
        except:
            self._fig, self.axeslist = plt.subplots(ncols=n+1, nrows=n+1)

        for ind in range(len(batch_x)):
            title = batch_pred[ind]
            self.axeslist.ravel()[ind].imshow(batch_x[ind,:,:,0], cmap=plt.jet())
            self.axeslist.ravel()[ind].set_title(title)
            self.axeslist.ravel()[ind].set_axis_off()
        plt.tight_layout() # optional
        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()
        # plt.imshow(batch_x[1][:,:,0])
        # plt.draw()
        plt.pause(1)


    def save(self, path, filename):
        """
        Saves the experimental results to ``results.json`` file
        :param path: path to results file
        :param filename: filename of results file
        """
        results_fn = os.path.join(path, filename)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(results_fn, "w") as f:
            json.dump(self.results, f, indent=4, sort_keys=True)
        f.close()

    def __del__(self):
        """
        Destructor of results list
        :return:
        """
        self.results.clear()

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
        self.b_location_out = tf.keras.layers.Dense(1,
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
        print('Glimpse:', glimpses)
        self.glimpses_list.append(glimpses[0])

        # Process glimpses
        print("Bandwidth:", self.sensorBandwidth)
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

class Decoder(tf.keras.Model):
    def __init__(self, units, batch_size, pixel_scaling, mnist_size,
            sensorBandwidth, totalSensorBandwidth, depth):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.units = units
        self.lstm = tf.keras.layers.LSTMCell(units, activation=tf.nn.relu, recurrent_initializer='zeros')

        # used for attention
        self.attention = AttentionControl(units, batch_size,
                pixel_scaling, mnist_size, sensorBandwidth, totalSensorBandwidth, depth)
        print('Decoder Initialized')

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.units))

    def reset_attention(self):
        self.attention.reset_lists()

    def get_attention_lists(self):
        return self.attention.get_lists()

    def call(self, inputs, hidden):
        # enc_output shape == (batch_size, max_length, hidden_size)
        glimpse = self.attention(tf.zeros((self.batch_size, self.units)), inputs)
        print('glimpse: ', glimpse)
        # passing the concatenated vector to the LSTM
        output, _ = self.lstm(glimpse, [hidden, hidden])
        print(output)
        glimpse = self.attention(output, inputs)
        print('glimpse: ', glimpse)

        # return output, glimpse 
        return glimpse 
