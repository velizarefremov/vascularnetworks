import tensorflow as tf
import numpy as np
from itkutilities import get_itk_array
import utility


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 5-D tensor: [batch_size, width, height, depth, channels]
    # Our input images are 256x256x256 pixels, and have one color channel
    input_layer = tf.reshape(features["x"], [-1, 64, 64, 64, 1])

    # Convolutional Layer #1
    # Computes 5 features using a 3x3x3 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 256, 256, 256, 1]
    # Output Tensor Shape: [batch_size, 256, 256, 256, 5]
    conv1 = tf.layers.conv3d(
        inputs=input_layer,
        filters=5,
        kernel_size=[3, 3, 3],
        padding="same",
        activation=tf.nn.relu)

    # Convolutional Layer #2
    # Computes 10 features using a 5x5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 256, 256, 256, 5]
    # Output Tensor Shape: [batch_size, 256, 256, 256, 10]
    conv2 = tf.layers.conv3d(
        inputs=conv1,
        filters=10,
        kernel_size=[5, 5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Convolutional Layer #3
    # Computes 20 features using a 5x5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 256, 256, 256, 10]
    # Output Tensor Shape: [batch_size, 256, 256, 256, 20]
    conv3 = tf.layers.conv3d(
        inputs=conv2,
        filters=20,
        kernel_size=[5, 5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Convolutional Layer #4
    # Computes 50 features using a 3x3x3 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 256, 256, 256, 20]
    # Output Tensor Shape: [batch_size, 256, 256, 256, 50]
    conv4 = tf.layers.conv3d(
        inputs=conv3,
        filters=50,
        kernel_size=[5, 5, 5],
        padding="same",
        activation=tf.nn.relu)


    # Convolutional Layer #5
    # Computes 2 features using a 3x3x3 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 256, 256, 256, 50]
    # Output Tensor Shape: [batch_size, 256, 256, 256, 2]
    logits = tf.layers.conv3d(
        inputs=conv4,
        filters=2,
        kernel_size=[3, 3, 3],
        padding="same",
        activation=tf.nn.relu)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=4),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    num_pos = tf.reduce_sum(labels, name='label_pos')
    num_neg = tf.subtract(204800, tf.reduce_sum(labels), name='label_neg')

    flat_probs_pos = tf.reshape(predictions['probabilities'][:, :, :, 1], [-1])
    flat_probs_neg = tf.reshape(predictions['probabilities'][:, :, :, 0], [-1])
    flat_labels_pos = tf.reshape(labels, [-1])
    flat_labels_neg = tf.add(tf.multiply(flat_labels_pos, -1), 1)

    pos_log_input = tf.add(tf.multiply(tf.to_float(flat_labels_pos), flat_probs_pos, name='mult_pos'),
                           tf.to_float(flat_labels_neg))
    neg_log_input = tf.add(tf.multiply(tf.to_float(flat_labels_neg), flat_probs_neg, name='mult_neg'),
                           tf.to_float(flat_labels_pos))

    log_probs_pos = tf.log(pos_log_input, name='log_pos')
    log_probs_neg = tf.log(neg_log_input, name='log_neg')

    part1 = tf.divide(tf.reduce_sum(log_probs_pos, name='reduce_pos_log'), tf.to_float(num_pos))
    part2 = tf.divide(tf.reduce_sum(log_probs_neg, name='reduce_neg_log'), tf.to_float(num_neg))
    loss = tf.add(tf.multiply(part1, -1), tf.multiply(part2, -1))

    # loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

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


def load_train_data_folder(folder_name, data_type=0, number_files=0):

    # Read info file.
    with open(folder_name + "info.txt") as f:
        content = f.readlines()

    content = [x.strip() for x in content]
    num_files = int(content[4])

    print("Total Number of Files: ", num_files)
    print("Loading training data...")

    if number_files != 0:
        num_files = number_files
        print("Loading only the first ", num_files, " of data.")

    if data_type == 0:
        data = np.zeros([num_files, 64, 64, 64], dtype='float32')
    else:
        data = np.zeros([num_files, 64, 64, 64], dtype='int32')

    for i in utility.my_range(0, num_files, 1):

        data_file_name = folder_name + "cropped" + str(i) + ".nii.gz"
        # 4 dimensional data
        # type == 0 is data, 1 is labeled
        if data_type == 0:
            data[i] = np.asarray(get_itk_array(data_file_name), dtype='float32')
        else:
            data[i] = np.asarray(get_itk_array(data_file_name), dtype='int32')

    print("All data loaded.")

    return data


def load_train_data(filename, type=0):
    """Loads training and/or evaluation data"""
    print("Loading training data...")

    # 4 dimensional data
    # type == 0 is data, 1 is labels
    if type == 0:
        data = np.zeros([1, 64, 64, 64], dtype=np.float32)
    else:
        data = np.zeros([1, 64, 64, 64], dtype=np.int32)

    data[0] = get_itk_array(filename)

    print("All data loaded.")
    return data


def load_test_data(filename):
    """Loads testing data"""
    print("Started loading test data.")

    data = np.zeros([1, 64, 64, 64], dtype=np.float32)

    data[0] = np.array(get_itk_array(filename))

    print("Loaded test data.")

    return data
