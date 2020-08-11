from MNIST_Processing import MNIST
#from networkTF2 import RAM
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

        self.lr_decay_type = PARAMETERS.LEARNING_RATE_DECAY_TYPE
        self.min_lr = PARAMETERS.MIN_LEARNING_RATE
        self.max_lr = PARAMETERS.LEARNING_RATE
        self.learning_rate = PARAMETERS.LEARNING_RATE
        self.lr_decay_rate = PARAMETERS.LEARNING_RATE_DECAY
        self.lr_decayt_steps = PARAMETERS.LEARNING_RATE_DECAY_STEPS
        self.momentum = PARAMETERS.MOMENTUM
        self.optimizer = PARAMETERS.OPTIMIZER
        self.batch_size = PARAMETERS.BATCH_SIZE
        self.max_epochs = PARAMETERS.MAX_EPOCHS
        self.M = DOMAIN_OPTIONS.MONTE_CARLO
        self.num_glimpses = DOMAIN_OPTIONS.NGLIMPSES
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


        #   ================
        #   Creating the RAM
        #   ================
        #   self.ram = RAM(totalSensorBandwidth, self.batch_size*self.M, PARAMETERS.OPTIMIZER, PARAMETERS.MOMENTUM,
        #         DOMAIN_OPTIONS.NGLIMPSES, pixel_scaling, mnist_size, DOMAIN_OPTIONS.CHANNELS, DOMAIN_OPTIONS.SCALING_FACTOR,
        #         DOMAIN_OPTIONS.SENSOR, DOMAIN_OPTIONS.DEPTH,
        #         PARAMETERS.LEARNING_RATE, PARAMETERS.LEARNING_RATE_DECAY, PARAMETERS.LEARNING_RATE_DECAY_STEPS, PARAMETERS.LEARNING_RATE_DECAY_TYPE,
        #         PARAMETERS.MIN_LEARNING_RATE)

        self.ram = Decoder(256, self.batch_size*self.M, pixel_scaling,
                mnist_size, DOMAIN_OPTIONS.SENSOR, totalSensorBandwidth,
                DOMAIN_OPTIONS.DEPTH, DOMAIN_OPTIONS.NGLIMPSES)


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

            _, pred_action, _= self.ram(X)

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

        """
        Function to control the linear decay
        of the learning rate
        :return: New learning rate
        """
        #if self.lr_decay_type == "linear":
        #    # Linear Learning Rate Decay
        #    def learning_rate_decay(epoch, lr):
        #        new_lr = max(self.min_lr, self.lr - self.lr_decay_rate)
        #        return new_lr
        #elif self.lr_decay_type == "exponential":
        #    # Exponential Learning Rate Decay
        #    def learning_rate_decay(epoch, lr):
        #        new_lr = max(self.min_lr, self.max_lr * (self.lr_decay_rate **
        #            epoch/self.lr_decay_steps))
        #        return new_lr
        #elif self.lr_decay_type == "exponential_staircase":
        #    # Exponential Learning Rate Decay
        #    def learning_rate_decay(epoch, lr):
        #        new_lr = max(self.min_lr, self.max_lr * (self.lr_decay_rate **
        #            (epoch // self.lr_decay_steps)))
        #        return new_lr
        #else:
        #    print("Wrong type of learning rate: " + self.lr_decay_type)
        #    def learning_rate_decay(self, epoch, lr):
        #        return new_lr

        #callback = tf.keras.callbacks.LearningRateScheduler(learning_rate_decay)
        #TODO: Wirte own lr Decay class
        # Choose Optimizer
        if self.optimizer == "rmsprop":
            trainer = tf.keras.optimizers.RMSProp(learning_rate=self.learning_rate)
        elif self.optimizer == "adam":
            trainer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer == "adadelta":
            trainer = tf.keras.optimizers.Adadelta(learning_rate=self.learning)
        elif self.optimizer == 'sgd':
            trainer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=self.momentum, nesterov=True)
        else:
            raise ValueError("unrecognized update: {}".format(self.optimizer))

        total_epochs = 0
        validation_accuracy = 0
        num_train_data = len(self.mnist.dataset.train._images)
        # Create file writer
        summary_writer = tf.summary.create_file_writer("summary")

        # Initial Performance Check
        performance_accuracy, performance_accuracy_std = self.performance_run(total_epochs)
        logging.info("Epoch={:d}: >>> Test-Accuracy: {:.4f} "
                       "+/- {:.6f}".format(total_epochs, performance_accuracy, performance_accuracy_std))
        tf.summary.scalar("Accuracy", performance_accuracy, step=total_epochs)
        summary_writer.flush()
        with summary_writer.as_default():
            patience_steps = 0
            early_stopping_accuracy = 0.
            visualize_classification = True
            logging.info("Start Training")
            for _ in range(self.max_epochs):
                start_time = time.time()
                train_accuracy = 0
                train_accuracy_sqrt = 0
                a_loss = []
                l_loss = []
                s_loss = []
                b_loss = []

                while total_epochs == self.mnist.dataset.train.epochs_completed:
                    X, Y, _= self.mnist.get_batch(self.batch_size, data_type="train")
                    with tf.GradientTape() as tape:
                        glimpse, pred, baseline = self.ram(X)
                        nnl_loss, reinforce_loss, reinforce_std_loss, baseline_loss, reward = self.ram.loss(Y, pred, baseline)
                        # TODO: Implement baseline
                        gradients_op_a = tape.gradient(nnl_loss, self.ram.variables)
                        #gradients_op_b = tape.gradient(b_loss, self.ram.baseline_layer)
                    trainer.apply_gradients(zip(gradients_op_a, self.ram.variables))
                    # trainer.apply_gradients(zip(gradients_op_b, self.ram.baseline_layer))

                    max_p_y = tf.argmax(pred, axis=-1)
                    train_accuracy += np.sum(np.equal(max_p_y, Y).astype(np.float32), axis=-1)
                    train_accuracy_sqrt+= np.sum((np.equal(max_p_y, Y).astype(np.float32))**2, axis=-1)
                    a_loss.append(nnl_loss)
                    l_loss.append(reinforce_loss)
                    s_loss.append(reinforce_std_loss)
                    b_loss.append(baseline_loss)

                # Get data for Tensorboard summary
                eval_location_list, location_list, location_mean_list, location_stddev_list, glimpses_list = self.ram.get_attention_lists()

                # take_first_zoom = []
                # for gl in range(self.glimpses):
                #     take_first_zoom.append(self.glimpses_list[gl][0])
                # self.summary_zooms = tf.summary.image("Zooms", tf.reshape(take_first_zoom, (self.glimpses, self.sensorBandwidth, self.sensorBandwidth, 1)), max_outputs=self.glimpses)

                total_epochs += 1


                # Train Accuracy
                train_accuracy = train_accuracy/(num_train_data*self.M)
                train_accuracy_std = np.sqrt(((train_accuracy_sqrt/(num_train_data*self.M)) - train_accuracy**2)/(num_train_data*self.M))

                # self.visualize(X[:self.batch_size],Y[:self.batch_size], Y[:self.batch_size])
                logging.info("Epoch={:d}: >>> Train-Accuracy: {:.4f} +/- {:.6f}".format(total_epochs, train_accuracy, train_accuracy_std))
                if total_epochs % 10 == 0:
                    # Test Accuracy
                    performance_accuracy, performance_accuracy_std = self.performance_run(total_epochs)

                    # Print out Infos
                    logging.info("Epoch={:d}: >>> Test-Accuracy: {:.4f} +/- {:.6f}".format(total_epochs, performance_accuracy, performance_accuracy_std))

            #        # Some visualization
            #        img, zooms = self.ram.get_images(np.vstack([self.test_images[0]]*self.batch_size*self.M))

            #        self.summary_writer.add_summary(img, total_epochs)
            #        self.summary_writer.add_summary(zooms, total_epochs)
            #        self.test_images = []
                else:
                    # Validation Accuracy
                    validation_accuracy, vaidation_accuracy_std = self.performance_run(total_epochs, validation=True)


                    # Print out Infos
                    logging.info("Epoch={:d}: >>> examples/s: {:.2f}, Accumulated-Loss: {:.4f}, Location-Mean Loss: {:.4f}, Location-Stddev Loss: {:.4f}, Baseline-Loss: {:.4f}, "
                                 "Learning Rate: {:.6f}, Train-Accuracy: {:.4f} +/- {:.6f}, "
                                 "Validation-Accuracy: {:.4f} +/- {:.6f}".format(total_epochs,
                                     float(num_train_data)/float(time.time()-start_time), np.mean(a_loss), np.mean(l_loss), np.mean(s_loss), np.mean(b_loss),
                                     lr, train_accuracy, train_accuracy_std, validation_accuracy, vaidation_accuracy_std))

                    # Early Stopping
                    if early_stopping and early_stopping_accuracy < validation_accuracy:
                        early_stopping_accuracy = validation_accuracy
                        patience_steps = 0
                    else:
                        patience_steps += 1

                # Gather information for Tensorboard
                tf.summary.scalar(name='Losses/Accumulated Loss',
                    data=float(np.mean(a_loss), step=total_epochs))
                tf.summary.scalar(name='Losses/Location: Mean Loss', data=float(np.mean(l_loss)), step=total_epochs)
                tf.summary.scalar(name='Losses/Location: Stddev Loss', data=float(np.mean(s_loss)), step=total_epochs)
                tf.summary.scalar(name='Losses/Baseline Loss', data=float(np.mean(b_loss)))
                tf.summary.scalar(name='Accuracy/Performance', data=float(performance_accuracy), step=total_epochs)
                tf.summary.scalar(name='Accuracy/Validation', data=float(validation_accuracy), step=total_epochs)
                tf.summary.scalar(name='Accuracy/Train', data=float(train_accuracy), step=total_epochs)

                summary_writer.flush()

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
                 self._fig.clear()
                 self._fig, self.axeslist = plt.subplots(ncols=n, nrows=n)
        except:
            self._fig, self.axeslist = plt.subplots(ncols=n, nrows=n)

        for ind in range(len(batch_x)):
            title = batch_pred[ind]
            self.axeslist.ravel()[ind].imshow(batch_x[ind,:,:,0], cmap=plt.jet())
            self.axeslist.ravel()[ind].set_title(title)
            self.axeslist.ravel()[ind].set_axis_off()

        # plt.tight_layout() # optional
        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()
        # plt.imshow(batch_x[1][:,:,0])
        # plt.draw()
        plt.pause(0.1)


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
        return self.eval_location_list, self.location_list, self.location_mean_list, self.location_stddev_list, self.glimpses_list

    def call(self, inputs, output):
        #TODO: First location has to be random
        mean_loc = hard_tanh( self.h_location_out(tf.stop_gradient(output)))
        std_loc = tf.nn.sigmoid( self.h_location_std_out(tf.stop_gradient(output)))
        # Clip location between [-1,1] and adjust its scale
        #sample_loc = hard_tanh(mean_loc + tf.cond(self.training,
        #    lambda: tf.random.normal(mean_loc.get_shape(), 0, std_loc), lambda: 0. ))
        sample_loc = hard_tanh(mean_loc + tf.random.normal(mean_loc.get_shape(), 0, std_loc))
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

class Decoder(tf.keras.Model):
    def __init__(self, units, batch_size, pixel_scaling, mnist_size,
            sensorBandwidth, totalSensorBandwidth, depth, num_glimpses):
        super(Decoder, self).__init__()
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

        # Use mean baseline of all glimpses
        #b_pred = []
        #for o in outputs:
        #    o = tf.reshape(o, (self.batch_size, self.units))
        #    b_pred.append(tf.squeeze(self.baseline_layer(o)))
        #    b = tf.transpose(tf.stack(b_pred),perm=[1,0])
        #b_ng = tf.stop_gradient(b)
        b_ng= tf.zeros((self.batch_size, 1))

        return glimpse, action_out, b_ng

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

        # Baseline is trained with MSE
        #b_loss = tf.keras.losses.MSE(R, baseline)
        b_loss = 0

        return cost, -Reinforce, -Reinforce_std, b_loss, reward

