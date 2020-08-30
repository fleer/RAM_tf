from MNIST_Processing import MNIST
from networkTF2 import RAM, Baseline
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
        self.lr_decay_steps = PARAMETERS.LEARNING_RATE_DECAY_STEPS
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

        self.ram = RAM(256, self.batch_size*self.M, pixel_scaling,
                mnist_size, DOMAIN_OPTIONS.SENSOR, totalSensorBandwidth,
                DOMAIN_OPTIONS.DEPTH, DOMAIN_OPTIONS.NGLIMPSES)

        # Learning the baseline independently
        self.baseline = Baseline(256, self.batch_size*self.M)

    #    self.saver = tf.train.Saver(max_to_keep=5)
    #    if PARAMETERS.LOAD_MODEL == True:
    #        print ('Loading Model...')
#   #            # ckpt = tf.train.get_checkpoint_state(PARAMETERS.MODEL_FILE_PATH)
    #        #self.saver.restore(sess, ckpt.model_checkpoint_path)
    #        self.saver.restore(sess, PARAMETERS.MODEL_FILE_PATH)
    #    else:
    #        sess.run(tf.global_variables_initializer())

        #   ================
        #   Train
        #   ================
        self.train(PARAMETERS.EARLY_STOPPING, PARAMETERS.PATIENCE)
        self.save('./', 'results.json')

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
        if self.lr_decay_type == "linear":
            # Linear Learning Rate Decay
            def learning_rate_decay(epoch, lr):
                new_lr = max(self.min_lr, self.lr - self.lr_decay_rate)
                return new_lr
        elif self.lr_decay_type == "exponential":
            # Exponential Learning Rate Decay
            def learning_rate_decay(epoch, lr):
                new_lr = max(self.min_lr, self.max_lr * (self.lr_decay_rate **
                    epoch/self.lr_decay_steps))
                return new_lr
        elif self.lr_decay_type == "exponential_staircase":
            # Exponential Learning Rate Decay
            def learning_rate_decay(epoch, lr):
                new_lr = max(self.min_lr, self.max_lr * (self.lr_decay_rate **
                    (epoch // self.lr_decay_steps)))
                return new_lr
        else:
            print("Wrong type of learning rate: " + self.lr_decay_type)
            def learning_rate_decay(self, epoch, lr):
                return new_lr

        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #     self.learning_rate,
        #     decay_steps=self.lr_decay_steps,
        #     decay_rate=self.lr_decay_rate,
        #     staircase=False)
        #callback = tf.keras.callbacks.LearningRateScheduler(learning_rate_decay)
        #TODO: Wirte own lr Decay class
        # Choose Optimizer
        if self.optimizer == "rmsprop":
            trainer = tf.keras.optimizers.RMSprop()
            trainer_b = tf.keras.optimizers.RMSprop()
        elif self.optimizer == "adam":
            trainer = tf.keras.optimizers.Adam()
            trainer_b = tf.keras.optimizers.Adam()
        elif self.optimizer == "adadelta":
            trainer = tf.keras.optimizers.Adadelta()
            trainer_b = tf.keras.optimizers.Adadelta()
        elif self.optimizer == 'sgd':
            trainer = tf.keras.optimizers.SGD(momentum=self.momentum, nesterov=True)
            trainer_b = tf.keras.optimizers.SGD(momentum=self.momentum, nesterov=True)
        else:
            raise ValueError("unrecognized update: {}".format(self.optimizer))

        total_epochs = 0
        validation_accuracy = 0
        num_train_data = len(self.mnist.dataset.train._images)
        # Create file writer
        summary_writer = tf.summary.create_file_writer("summary")

        with summary_writer.as_default():
            # Initial Performance Check
            # performance_accuracy, performance_accuracy_std = self.performance_run(total_epochs)
            # logging.info("Epoch={:d}: >>> Test-Accuracy: {:.4f} "
            #                "+/- {:.6f}".format(total_epochs, performance_accuracy, performance_accuracy_std))
            # tf.summary.scalar("Accuracy", performance_accuracy, step=total_epochs)
            summary_writer.flush()
            patience_steps = 0
            early_stopping_accuracy = 0.
            visualize_classification = True
            logging.info("Start Training")
            step = 0
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
                        tape.watch(self.ram.trainable_weights)
                        glimpse, pred, outputs = self.ram(X)
                        with tf.GradientTape() as tape_b:
                            tape_b.watch(self.baseline.trainable_weights)
                            baseline = self.baseline(outputs)
                            # TODO: Implement baseline
                            nnl_loss, reinforce_loss, reinforce_std_loss, R = self.ram.loss(Y, pred, baseline)
                            # Baseline is trained with MSE
                            baseline_loss = tf.keras.losses.mean_squared_error(R, baseline)

                            # print("nnl_loss ", nnl_loss)
                            # print("reinforce_loss", reinforce_loss)
                            # print("reinforce_std_loss", reinforce_std_loss)
                            # print("baseline_loss", baseline_loss)
                        gradients_op_b = tape_b.gradient(baseline_loss, self.baseline.trainable_weights)
                    gradients_op_a = tape.gradient(nnl_loss, self.ram.trainable_weights)
                    trainer.apply_gradients(zip(gradients_op_a, self.ram.trainable_weights))
                    trainer_b.apply_gradients(zip(gradients_op_b, self.baseline.trainable_weights))
                    max_p_y = tf.argmax(pred, axis=-1)
                    train_accuracy += np.sum(np.equal(max_p_y, Y).astype(np.float32), axis=-1)
                    train_accuracy_sqrt+= np.sum((np.equal(max_p_y, Y).astype(np.float32))**2, axis=-1)
                    a_loss.append(nnl_loss)
                    l_loss.append(reinforce_loss)
                    s_loss.append(reinforce_std_loss)
                    b_loss.append(baseline_loss)
                    # print("nnl_loss ", nnl_loss)
                    # print("reinforce_loss", reinforce_loss)
                    # print("reinforce_std_loss", reinforce_std_loss)
                    # print("baseline_loss", baseline_loss)
                    # step += 1
                    # self.learning_rate = learning_rate_decay(step, self.learning_rate)
                    # print(gradients_op_a)
                    # print(gradients_op_b)
                    # print('learning rate:', self.learning_rate)
                    trainer.learning_rate = self.learning_rate
                    trainer_b.learning_rate = self.learning_rate


                # Get data for Tensorboard summary
                eval_location_list, location_list, location_mean_list, location_stddev_list, glimpses_list = self.ram.get_attention_lists()

                # take_first_zoom = []
                # for gl in range(self.glimpses):
                #     take_first_zoom.append(self.glimpses_list[gl][0])
                # self.summary_zooms = tf.summary.image("Zooms", tf.reshape(take_first_zoom, (self.glimpses, self.sensorBandwidth, self.sensorBandwidth, 1)), max_outputs=self.glimpses)

                self.learning_rate = learning_rate_decay(total_epochs, self.learning_rate)
                total_epochs += 1


                # Train Accuracy
                train_accuracy = train_accuracy/(num_train_data*self.M)
                train_accuracy_std = np.sqrt(((train_accuracy_sqrt/(num_train_data*self.M)) - train_accuracy**2)/(num_train_data*self.M))

                # self.visualize(X[:self.batch_size],Y[:self.batch_size], Y[:self.batch_size])
                logging.info("Epoch={:d}: >>> Train-Accuracy: {:.4f} +/- {:.6f}, Learning Rate: {:.6f}".format(total_epochs, train_accuracy, train_accuracy_std, self.learning_rate))
                if total_epochs % 10 == 0:
                    # Test Accuracy
                    performance_accuracy, performance_accuracy_std = self.performance_run(total_epochs)

                    # Print out Infos
                    logging.info("Epoch={:d}: >>> Test-Accuracy: {:.4f} +/- {:.6f}".format(total_epochs, performance_accuracy, performance_accuracy_std))
                    tf.summary.scalar(name='Accuracy/Performance', data=float(performance_accuracy), step=total_epochs)
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
                            "Train-Accuracy: {:.4f} +/- {:.6f}, "
                                 "Validation-Accuracy: {:.4f} +/- {:.6f}".format(total_epochs,
                                     float(num_train_data)/float(time.time()-start_time), np.mean(a_loss), np.mean(l_loss), np.mean(s_loss), np.mean(b_loss), train_accuracy, train_accuracy_std, validation_accuracy, vaidation_accuracy_std))
                    tf.summary.scalar(name='Accuracy/Validation', data=float(validation_accuracy), step=total_epochs)

                    # Early Stopping
                    if early_stopping and early_stopping_accuracy < validation_accuracy:
                        early_stopping_accuracy = validation_accuracy
                        patience_steps = 0
                    else:
                        patience_steps += 1

                # Gather information for Tensorboard
                tf.summary.scalar(name='Losses/Accumulated Loss', data=float(np.mean(a_loss)), step=total_epochs)
                tf.summary.scalar(name='Losses/Location: Mean Loss', data=float(np.mean(l_loss)), step=total_epochs)
                tf.summary.scalar(name='Losses/Location: Stddev Loss', data=float(np.mean(s_loss)), step=total_epochs)
                tf.summary.scalar(name='Losses/Baseline Loss', data=float(np.mean(b_loss)), step=total_epochs)
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

