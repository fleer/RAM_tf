from MNIST_Processing import MNIST
from network import RAM
import numpy as np
import tensorflow as tf
from collections import defaultdict
import logging
import time
import os
import sys
import json

class Experiment():
    """
    Main class, controlling the experiment
    """

    results = defaultdict(list)


    def __init__(self, PARAMETERS, DOMAIN_OPTIONS, results_file="results.json", model_file="network.h5"):

        logging.basicConfig(level=logging.INFO)

        #   ================
        #   Reading the parameters
        #   ================

        mnist_size = DOMAIN_OPTIONS.MNIST_SIZE
        channels = DOMAIN_OPTIONS.CHANNELS
        scaling_factor = DOMAIN_OPTIONS.SCALING_FACTOR
        sensorResolution = DOMAIN_OPTIONS.SENSOR
        self.loc_std = DOMAIN_OPTIONS.LOC_STD
        self.nZooms = DOMAIN_OPTIONS.DEPTH
        self.nGlimpses = DOMAIN_OPTIONS.NGLIMPSES

        self.batch_size = PARAMETERS.BATCH_SIZE
        self.max_epochs = PARAMETERS.MAX_EPOCHS

        totalSensorBandwidth = self.nZooms * sensorResolution * sensorResolution * channels

        #   ================
        #   Loading the MNIST Dataset
        #   ================

        self.mnist = MNIST(mnist_size, self.batch_size, DOMAIN_OPTIONS.TRANSLATE, DOMAIN_OPTIONS.TRANSLATED_MNIST_SIZE)

        #   ================
        #   Creating the RAM
        #   ================

        if DOMAIN_OPTIONS.TRANSLATE:
            pixel_scaling = (DOMAIN_OPTIONS.UNIT_PIXELS * 2.)/ float(DOMAIN_OPTIONS.TRANSLATED_MNIST_SIZE)
        else:
            pixel_scaling = (DOMAIN_OPTIONS.UNIT_PIXELS * 2.)/ float(DOMAIN_OPTIONS.MNIST_SIZE)


        tf.reset_default_graph()
        self.summary_writer = tf.summary.FileWriter("summary")


        with tf.Session() as sess:
            #   ================
            #   Train
            #   ================
            self.ram = RAM(totalSensorBandwidth, self.batch_size, self.nGlimpses, pixel_scaling,
                           PARAMETERS.LEARNING_RATE, PARAMETERS.LEARNING_RATE_DECAY,
                           PARAMETERS.MIN_LEARNING_RATE, 2*DOMAIN_OPTIONS.LOC_STD, sess)

            sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver(max_to_keep=5)

            self.train(PARAMETERS.LEARNING_RATE, PARAMETERS.LEARNING_RATE_DECAY, PARAMETERS.EARLY_STOPPING, PARAMETERS.PATIENCE, sess)
        self.save('./', results_file)

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

        for i in range(batches_in_epoch):
            if validation:
                X, Y= self.mnist.get_batch_validation(self.batch_size)
            else:
                X, Y= self.mnist.get_batch_test(self.batch_size)

            X = np.reshape(X, (self.batch_size, 28, 28, 1))
            _, pred_action = self.ram.evaluate(X,Y)
            actions += np.sum(np.equal(pred_action,Y).astype(np.float32), axis=-1)
            actions_sqrt += np.sum((np.equal(pred_action,Y).astype(np.float32))**2, axis=-1)

        accuracy = actions/num_data
        accuracy_std = np.sqrt(((actions_sqrt/num_data) - accuracy**2)/num_data)

        if not validation:
            # Save to results file
            self.results['learning_steps'].append(total_epochs)
            self.results['accuracy'].append(accuracy)
            self.results['accuracy_std'].append(accuracy_std)

        return accuracy, accuracy_std

    def train(self, lr, lr_decay, early_stopping, patience, session):
        """
        Training the current model
        :param lr: learning rate
        :param lr_decay: Number of steps the Learning rate should (linearly)
        :param early_stopping: Use early stopping
        :param patience: Number of Epochs observing the worsening of
                Validation set, before stopping
        :return:
        """
        total_epochs = 0
        # Initial Performance Check
        #accuracy, accuracy_std = self.performance_run(total_epochs)
        #logging.info("Epoch={:d}: >>> Test-Accuracy: {:.4f} "
        #              "+/- {:.6f}".format(total_epochs, accuracy, accuracy_std))
        num_train_data = len(self.mnist.dataset.train._images)
        summary = tf.Summary()

        patience_steps = 0
        early_stopping_accuracy = 0.
        for i in range(self.max_epochs):
            start_time = time.time()
            test_accuracy = 0
            test_accuracy_sqrt = 0
            a_loss = []
            l_loss = []
            b_loss = []
            while total_epochs == self.mnist.dataset.train.epochs_completed:
                X, Y= self.mnist.get_batch_train(self.batch_size)
                X = np.reshape(X, (self.batch_size, 28, 28, 1))
                _, pred_action, nnl_loss, reinforce_loss, baseline_loss = self.ram.train(X,Y)
                test_accuracy += np.sum(np.equal(pred_action,Y).astype(np.float32), axis=-1)
                test_accuracy_sqrt+= np.sum((np.equal(pred_action,Y).astype(np.float32))**2, axis=-1)
                a_loss.append(nnl_loss)
                l_loss.append(reinforce_loss)
                b_loss.append(baseline_loss)
            total_epochs += 1
            if lr_decay > 0:
                lr = self.ram.learning_rate_decay()


            summary.value.add(tag='Losses/Action Loss', simple_value=float(np.mean(a_loss)))
            summary.value.add(tag='Losses/Location Loss', simple_value=float(np.mean(l_loss)))
            summary.value.add(tag='Losses/Baseline Loss', simple_value=float(np.mean(b_loss)))


            if total_epochs % 10 == 0:
                # Test Accuracy
                accuracy, accuracy_std = self.performance_run(total_epochs)
                summary.value.add(tag='Performance/Accuracy', simple_value=float(accuracy))

                # Print out Infos
                logging.info("Epoch={:d}: >>> Test-Accuracy: {:.4f} +/- {:.6f}".format(total_epochs, accuracy, accuracy_std))
            else:
                # Validation Accuracy
                accuracy, accuracy_std = self.performance_run(total_epochs, validation=True)
                summary.value.add(tag='Validation/Accuracy', simple_value=float(accuracy))

                # Test Accuracy
                test_accuracy = test_accuracy/num_train_data
                test_accuracy_std = np.sqrt(((test_accuracy_sqrt/num_train_data) - test_accuracy**2)/num_train_data)
                summary.value.add(tag='Train/Accuracy', simple_value=float(test_accuracy))

                # Print out Infos
                logging.info("Epoch={:d}: >>> examples/s: {:.2f}, Action-Loss: {:.4f}, Location-Loss: {:.4f}, Baseline-Loss: {:.4f}, "
                             "Learning Rate: {:.6f}, Train-Accuracy: {:.4f} +/- {:.6f}, "
                             "Validation-Accuracy: {:.4f} +/- {:.6f}".format(total_epochs,
                                 float(num_train_data)/float(time.time()-start_time), np.mean(a_loss), np.mean(l_loss), np.mean(b_loss),
                                 lr, test_accuracy, test_accuracy_std, accuracy, accuracy_std))

                # Early Stopping
                if early_stopping and early_stopping_accuracy < accuracy:
                    early_stopping_accuracy = accuracy
                    patience_steps = 0
                else:
                    patience_steps += 1

            self.summary_writer.add_summary(summary, total_epochs)

            self.summary_writer.flush()

            if patience_steps > patience:
                self.saver.save(session, './Model/best_model-' + str(total_epochs) + '.cptk')
                logging.info("Early Stopping at Epoch={:d}! Validation Accuracy is not increasing. The best Newtork will be saved!".format(total_epochs))
                return 0

            if total_epochs % 100 == 0:
                self.saver.save(session, './Model/model-' + str(total_epochs) + '.cptk')

    def save(self, path, filename):
        """
        Saves the experimental results to ``results.json`` file
        """
        results_fn = os.path.join(path, filename)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(results_fn, "w") as f:
            json.dump(self.results, f, indent=4, sort_keys=True)
        f.close()

    def __del__(self):
        self.results.clear()
