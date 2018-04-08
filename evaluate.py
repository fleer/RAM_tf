from MNIST_Processing import MNIST
from network import RAM
import numpy as np
import tensorflow as tf
import sys
import cv2
from matplotlib import pyplot as plt

# This is not a nice way to implement the different configuration scripts...
if len(sys.argv) > 1:
    if sys.argv[1] == 'run_mnist':
        from run_mnist import MNIST_DOMAIN_OPTIONS
        from run_mnist import PARAMETERS
    elif sys.argv[1] == 'run_translated_mnist':
        from run_translated_mnist import MNIST_DOMAIN_OPTIONS
        from run_translated_mnist import PARAMETERS
    else:
        print "Wrong file name for confiuration file!"
        sys.exit(0)
else:
    print "Give Configuration File as additional argument! \n " \
          "E.g. python evaluate.py run_mnist ./model/network.h5"
    sys.exit(0)

# Save the pictures
save = True

#   ================
#   Reading the parameters
#   ================

channels = MNIST_DOMAIN_OPTIONS.CHANNELS
sensorResolution = MNIST_DOMAIN_OPTIONS.SENSOR
scaling = MNIST_DOMAIN_OPTIONS.SCALING_FACTOR
loc_std = MNIST_DOMAIN_OPTIONS.LOC_STD
nZooms = MNIST_DOMAIN_OPTIONS.DEPTH
nGlimpses = MNIST_DOMAIN_OPTIONS.NGLIMPSES

#batch_size = PARAMETERS.BATCH_SIZE
batch_size = 4
max_epochs = PARAMETERS.MAX_EPOCHS



totalSensorBandwidth = nZooms * sensorResolution * sensorResolution * channels

#   ================
#   Loading the MNIST Dataset
#   ================

mnist = MNIST(MNIST_DOMAIN_OPTIONS.MNIST_SIZE, batch_size, MNIST_DOMAIN_OPTIONS.TRANSLATE, MNIST_DOMAIN_OPTIONS.TRANSLATED_MNIST_SIZE, 1)


if MNIST_DOMAIN_OPTIONS.TRANSLATE:
    pixel_scaling = (MNIST_DOMAIN_OPTIONS.UNIT_PIXELS * 2.)/ float(MNIST_DOMAIN_OPTIONS.TRANSLATED_MNIST_SIZE)
else:
    pixel_scaling = (MNIST_DOMAIN_OPTIONS.UNIT_PIXELS * 2.)/ float(MNIST_DOMAIN_OPTIONS.MNIST_SIZE)

if MNIST_DOMAIN_OPTIONS.TRANSLATE:
    mnist_size = MNIST_DOMAIN_OPTIONS.TRANSLATED_MNIST_SIZE
else:
    mnist_size = MNIST_DOMAIN_OPTIONS.MNIST_SIZE

def pad_to_bounding_box(image, offset_height, offset_width, target_height,
                        target_width):
    """Pad `image` with zeros to the specified `height` and `width`.
    Adds `offset_height` rows of zeros on top, `offset_width` columns of
    zeros on the left, and then pads the image on the bottom and right
    with zeros until it has dimensions `target_height`, `target_width`.
    This op does nothing if `offset_*` is zero and the image already has size
    `target_height` by `target_width`.
    Args:
      image: 4-D Tensor of shape `[batch, height, width, channels]` or
             3-D Tensor of shape `[height, width, channels]`.
      offset_height: Number of rows of zeros to add on top.
      offset_width: Number of columns of zeros to add on the left.
      target_height: Height of output image.
      target_width: Width of output image.
    Returns:
      If `image` was 4-D, a 4-D float Tensor of shape
      `[batch, target_height, target_width, channels]`
      If `image` was 3-D, a 3-D float Tensor of shape
      `[target_height, target_width, channels]`
    Raises:
      ValueError: If the shape of `image` is incompatible with the `offset_*` or
        `target_*` arguments, or either `offset_height` or `offset_width` is
        negative.
    """

    is_batch = True
    image_shape = image.shape
    if image.ndim == 3:
        is_batch = False
        image = np.expand_dims(image, 0)
    elif image.ndim is None:
        is_batch = False
        image = np.expand_dims(image, 0)
        image.set_shape([None] * 4)
    elif image_shape.ndims != 4:
        raise ValueError('\'image\' must have either 3 or 4 dimensions.')

    batch = len(image)
    height = len(image[0])
    width = len(image[0,0])
    depth = len(image[0,0,0])

    after_padding_width = target_width - offset_width - width
    after_padding_height = target_height - offset_height - height

    assert offset_height >= 0, 'offset_height must be >= 0'
    assert offset_width >= 0, 'offset_width must be >= 0'
    assert after_padding_width >= 0, 'width must be <= target - offset'
    assert after_padding_height >= 0, 'height must be <= target - offset'

    # Do not pad on the depth dimensions.
    paddings = np.reshape(
        np.stack([
            0, 0, offset_height, after_padding_height, offset_width,
            after_padding_width, 0, 0
        ]), [4, 2])
    padded = np.pad(image, paddings, 'constant', constant_values=0)

    padded_shape = [i for i in [batch, target_height, target_width, depth]]
    np.reshape(padded, padded_shape)

    if not is_batch:
        padded = np.squeeze(padded, axis=0)

    return padded

def glimpseSensor(img, normLoc):
    assert not np.any(np.isnan(normLoc))," Locations have to be between 1, -1: {}".format(normLoc)
    assert np.any(np.abs(normLoc)<=1)," Locations have to be between 1, -1: {}".format(normLoc)


    # Convert location [-1,1] into MNIST Coordinates:
    loc = np.around(((normLoc + 1) / 2.) * mnist_size)
    loc = loc.astype(np.int32)

    img = np.reshape(img, (batch_size, mnist_size, mnist_size, channels))

    zooms = []

    # process each image individually
    for k in range(batch_size):
        imgZooms = []
        one_img = img[k,:,:,:]
        offset = sensorResolution* (scaling ** (nZooms-1))

        # pad image with zeros
        one_img = pad_to_bounding_box(one_img, offset, offset, \
                                           2*offset + mnist_size, 2*offset + mnist_size)

        for i in range(nZooms):
            d = int(sensorResolution * (scaling ** i))
            r = d//2

            loc_k = loc[k,:]
            adjusted_loc = offset + loc_k - r

            one_img2 = np.reshape(one_img, (one_img.shape[0], \
                                            one_img.shape[1]))

            # crop image to (d x d)
            zoom = one_img2[adjusted_loc[0]:adjusted_loc[0]+d, adjusted_loc[1]:adjusted_loc[1]+d]
            assert not np.any(np.equal(zoom.shape, (0,0))), "Picture has size 0, location {}, depth {}".format(adjusted_loc, d)
            assert len(zoom[0]) == d and len(zoom[1]) == d, "Glimpse has the dims: {}".format(zoom.shape)

            # resize cropped image to (sensorBandwidth x sensorBandwidth)
            if i > 0:
                zoom = cv2.resize(zoom, (sensorResolution, sensorResolution),
                                  interpolation=cv2.INTER_LINEAR)
            #zoom = np.reshape(zoom, (self.sensorBandwidth, self.sensorBandwidth))
            imgZooms.append(zoom)
        zooms.append(np.stack(imgZooms))

    shapes = set(arr.shape for arr in zooms)
    assert len(shapes) == 1, "zooms have different shapes: {}".format(zooms)
    zooms = np.stack(zooms)

    return zooms


tf.reset_default_graph()


with tf.Session() as sess:

    #   ================
    #   Creating the RAM
    #   ================
    ram = RAM(totalSensorBandwidth, batch_size, PARAMETERS.OPTIMIZER, PARAMETERS.MOMENTUM, nGlimpses, pixel_scaling, mnist_size, MNIST_DOMAIN_OPTIONS.CHANNELS, MNIST_DOMAIN_OPTIONS.SCALING_FACTOR,
                   MNIST_DOMAIN_OPTIONS.SENSOR, MNIST_DOMAIN_OPTIONS.DEPTH,
                   PARAMETERS.LEARNING_RATE, PARAMETERS.LEARNING_RATE_DECAY, PARAMETERS.LEARNING_RATE_DECAY_STEPS, PARAMETERS.LEARNING_RATE_DECAY_TYPE,
                   PARAMETERS.MIN_LEARNING_RATE, MNIST_DOMAIN_OPTIONS.LOC_STD, sess)

    saver = tf.train.Saver(max_to_keep=5)
    if len(sys.argv) > 2:
        print ('Loading Model...')
        try:
            ckpt = tf.train.get_checkpoint_state('./' + sys.argv[2])
            saver.restore(sess, ckpt.model_checkpoint_path)
            #saver.restore(sess, './', sys.argv[2])
            print("Loaded wights from " + sys.argv[2] + "!")
        except:
            print("Weights from " + sys.argv[2] +
                  " could not be loaded!")
            sys.exit(0)
    else:
        print("No weight file provided! New model initialized!")
        sess.run(tf.global_variables_initializer())


    plt.ion()
    plt.show()

    if MNIST_DOMAIN_OPTIONS.TRANSLATE:
        mnist_size = MNIST_DOMAIN_OPTIONS.TRANSLATED_MNIST_SIZE

    X, Y= mnist.get_batch(batch_size, "test")
    img = np.reshape(X, (batch_size, mnist_size, mnist_size, channels))
    for k in range(batch_size):
        one_img = img[k,:,:,:]

        plt.title(Y[k], fontsize=40)
        plt.imshow(one_img[:,:,0], cmap=plt.get_cmap('gray'),
                   interpolation="nearest")
        plt.draw()
        #time.sleep(0.05)
        if save:
            plt.savefig("symbol_" + repr(k) + ".png")
        plt.pause(.25)

    feed_dict = {ram.inputs_placeholder: X, ram.actions: Y, ram.training: False}#,
    fetches = [ram.reward, ram.predicted_probs, ram.eval_location_list]
    reward_fetched, predicted_labels_fetched, loc_list = sess.run(fetches, feed_dict=feed_dict)

    for n in range(nGlimpses):
        zooms = glimpseSensor(X,loc_list[n])
        ng = 1
        for g in range(batch_size):
            nz = 1
            plt.title(Y[g], fontsize=40)
            for z in zooms[g]:
                plt.imshow(z[:,:], cmap=plt.get_cmap('gray'),
                           interpolation="nearest")

                plt.draw()
                if save:
                    plt.savefig("symbol_" + repr(g) + "_" +
                                "glimpse_" + repr(n) + "_" +
                                "zoom_" + repr(nz) + ".png")
                #time.sleep(0.05)
                plt.pause(.25)
                nz += 1
            ng += 1


