from MNIST_Processing import MNIST
from network import RAM
import numpy as np
import tensorflow as tf
from collections import defaultdict
import logging
import time
import os
import json

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
        self.test_images = []
        if DOMAIN_OPTIONS.TRANSLATE:
            pixel_scaling = (DOMAIN_OPTIONS.UNIT_PIXELS * 2.)/ float(DOMAIN_OPTIONS.TRANSLATED_MNIST_SIZE)
        else:
            pixel_scaling = (DOMAIN_OPTIONS.UNIT_PIXELS * 2.)/ float(DOMAIN_OPTIONS.MNIST_SIZE)

        if DOMAIN_OPTIONS.TRANSLATE:
            mnist_size = DOMAIN_OPTIONS.TRANSLATED_MNIST_SIZE
        else:
            mnist_size = DOMAIN_OPTIONS.MNIST_SIZE

        totalSensorBandwidth = DOMAIN_OPTIONS.DEPTH * DOMAIN_OPTIONS.SENSOR * DOMAIN_OPTIONS.SENSOR * DOMAIN_OPTIONS.CHANNELS

        #   ================
        #   Loading the MNIST Dataset
        #   ================

        self.mnist = MNIST(DOMAIN_OPTIONS.MNIST_SIZE, self.batch_size, DOMAIN_OPTIONS.TRANSLATE, DOMAIN_OPTIONS.TRANSLATED_MNIST_SIZE)

        tf.reset_default_graph()
        self.summary_writer = tf.summary.FileWriter("summary")

        with tf.Session() as sess:

            #   ================
            #   Creating the RAM
            #   ================
            self.ram = RAM(totalSensorBandwidth, self.batch_size, PARAMETERS.OPTIMIZER, PARAMETERS.MOMENTUM, DOMAIN_OPTIONS.NGLIMPSES, pixel_scaling, mnist_size, DOMAIN_OPTIONS.CHANNELS, DOMAIN_OPTIONS.SCALING_FACTOR,
                           DOMAIN_OPTIONS.SENSOR, DOMAIN_OPTIONS.DEPTH,
                           PARAMETERS.LEARNING_RATE, PARAMETERS.LEARNING_RATE_DECAY, PARAMETERS.LEARNING_RATE_DECAY_STEPS, PARAMETERS.LEARNING_RATE_DECAY_TYPE,
                           PARAMETERS.MIN_LEARNING_RATE, DOMAIN_OPTIONS.LOC_STD, sess)

            self.saver = tf.train.Saver(max_to_keep=5)
            if PARAMETERS.LOAD_MODEL == True:
                print ('Loading Model...')
#               # ckpt = tf.train.get_checkpoint_state(PARAMETERS.MODEL_FILE_PATH)
                #self.saver.restore(sess, ckpt.model_checkpoint_path)
                self.saver.restore(sess, PARAMETERS.MODEL_FILE_PATH)
            else:
                sess.run(tf.global_variables_initializer())

            #   ================
            #   Train
            #   ================
            self.train(PARAMETERS.EARLY_STOPPING, PARAMETERS.PATIENCE, sess)
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

        for i in range(batches_in_epoch):
            if validation:
                X, Y= self.mnist.get_batch_validation(self.batch_size)
            else:
                X, Y= self.mnist.get_batch_test(self.batch_size)
                self.test_images = X

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

    def train(self, early_stopping, patience, session):
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
        performance_accuracy, performance_accuracy_std = self.performance_run(total_epochs)
        logging.info("Epoch={:d}: >>> Test-Accuracy: {:.4f} "
                      "+/- {:.6f}".format(total_epochs, performance_accuracy, performance_accuracy_std))
        num_train_data = len(self.mnist.dataset.train._images)

        patience_steps = 0
        early_stopping_accuracy = 0.
        for i in range(self.max_epochs):
            summary = tf.Summary()
            start_time = time.time()
            train_accuracy = 0
            train_accuracy_sqrt = 0
            a_loss = []
            l_loss = []
            b_loss = []
            while total_epochs == self.mnist.dataset.train.epochs_completed:
                X, Y= self.mnist.get_batch_train(self.batch_size)
                _, pred_action, nnl_loss, reinforce_loss, baseline_loss = self.ram.train(X,Y)
                train_accuracy += np.sum(np.equal(pred_action,Y).astype(np.float32), axis=-1)
                train_accuracy_sqrt+= np.sum((np.equal(pred_action,Y).astype(np.float32))**2, axis=-1)
                a_loss.append(nnl_loss)
                l_loss.append(reinforce_loss)
                b_loss.append(baseline_loss)
            total_epochs += 1
            lr = self.ram.learning_rate_decay()

            # Train Accuracy
            train_accuracy = train_accuracy/num_train_data

            if total_epochs % 10 == 0:
                # Test Accuracy
                performance_accuracy, performance_accuracy_std = self.performance_run(total_epochs)

                # Print out Infos
                logging.info("Epoch={:d}: >>> Test-Accuracy: {:.4f} +/- {:.6f}".format(total_epochs, performance_accuracy, performance_accuracy_std))

                # Some visualization
                img, zooms = self.ram.get_images(np.vstack([self.test_images[0]]*self.batch_size))

                self.summary_writer.add_summary(img, total_epochs)
                self.summary_writer.add_summary(zooms, total_epochs)
                self.test_images = []
            else:
                # Validation Accuracy
                validation_accuracy, vaidation_accuracy_std = self.performance_run(total_epochs, validation=True)

                train_accuracy_std = np.sqrt(((train_accuracy_sqrt/num_train_data) - train_accuracy**2)/num_train_data)

                # Print out Infos
                logging.info("Epoch={:d}: >>> examples/s: {:.2f}, Action-Loss: {:.4f}, Location-Loss: {:.4f}, Baseline-Loss: {:.4f}, "
                             "Learning Rate: {:.6f}, Train-Accuracy: {:.4f} +/- {:.6f}, "
                             "Validation-Accuracy: {:.4f} +/- {:.6f}".format(total_epochs,
                                 float(num_train_data)/float(time.time()-start_time), np.mean(a_loss), np.mean(l_loss), np.mean(b_loss),
                                 lr, train_accuracy, train_accuracy_std, validation_accuracy, vaidation_accuracy_std))

                # Early Stopping
                if early_stopping and early_stopping_accuracy < validation_accuracy:
                    early_stopping_accuracy = validation_accuracy
                    patience_steps = 0
                else:
                    patience_steps += 1

            # Gather information for Tensorboard
            summary.value.add(tag='Losses/Accumulated Loss', simple_value=float(np.mean(a_loss)))
            summary.value.add(tag='Losses/Location Loss', simple_value=float(np.mean(l_loss)))
            summary.value.add(tag='Losses/Baseline Loss', simple_value=float(np.mean(b_loss)))
            summary.value.add(tag='Accuracy/Performance', simple_value=float(performance_accuracy))
            summary.value.add(tag='Accuracy/Validation', simple_value=float(validation_accuracy))
            summary.value.add(tag='Accuracy/Train', simple_value=float(train_accuracy))

            self.summary_writer.add_summary(summary, total_epochs)

            self.summary_writer.flush()

            # Early Stopping
            if patience_steps > patience:
                self.saver.save(session, './Model/best_model-' + str(total_epochs) + '.cptk')
                logging.info("Early Stopping at Epoch={:d}! Validation Accuracy is not increasing. The best Newtork will be saved!".format(total_epochs))
                return 0

            # Save Model
            if total_epochs % 100 == 0:
                self.saver.save(session, save_path='./Model', global_step=total_epochs)

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
