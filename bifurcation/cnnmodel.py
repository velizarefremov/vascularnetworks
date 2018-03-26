import tensorflow as tf
import numpy as np
import utility
from itkutilities import get_itk_array


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 5-D tensor: [batch_size, width, height, depth, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    input_layer = tf.reshape(features["x"], [-1, 17, 17, 17, 1])

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    conv1 = tf.layers.conv3d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=[2, 2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    conv2 = tf.layers.conv3d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2, 2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 4 * 4 * 4 * 64])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(inputs=dropout, units=2)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"]),
        "precision": tf.metrics.precision(labels=labels, predictions=predictions["classes"]),
        "recall": tf.metrics.recall(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def load_train_data(count=2000, start=10000):
    """Loads training and/or evaluation data"""
    print("Loading data...")

    biffolder = '..\\..\\data\\bifs'
    nobiffolder = '..\\..\\data\\nobifs'

    # 4 dimensional data
    data = np.zeros([2 * count, 17, 17, 17], dtype=np.float32)

    # print(np.shape(data))

    # First load the bif data.
    for i in utility.my_range(start, start + count, 1):
        currentfile = biffolder + '\\cropped' + str(i) + ".nii.gz"
        data[i - start, :, :, :] = np.array(get_itk_array(currentfile))

    print("Loaded bif data.")

    # Now load the no bif data.
    for i in utility.my_range(start, start + count, 1):
        currentfile = nobiffolder + '\\cropped' + str(i) + ".nii.gz"
        data[i + count - start, :, :, :] = np.array(get_itk_array(currentfile))

    print("Loaded no bif data.")
    print("All data loaded.")
    return data


def load_test_data(filename, blocksize=8):
    """Loads testing data"""
    print("Started loading.")

    inputimage = np.array(get_itk_array(filename))

    inputsize = np.shape(inputimage)
    xsize = inputsize[0]
    ysize = inputsize[1]
    zsize = inputsize[2]

    count = 128*128*128

    data = np.zeros([count, 17, 17, 17], dtype=np.float32)

    index = 0

    for i in utility.my_range(blocksize, blocksize + 128, 1):

        # print("Step", i)

        for j in utility.my_range(blocksize, blocksize + 128, 1):

            for k in utility.my_range(blocksize, blocksize + 128, 1):

                data[index, :, :, :] = inputimage[(i - blocksize):(i + blocksize + 1),
                                       (j - blocksize):(j + blocksize + 1),
                                       (k - blocksize):(k + blocksize + 1)]
                index = index + 1

    print("Loaded data.")

    return data
