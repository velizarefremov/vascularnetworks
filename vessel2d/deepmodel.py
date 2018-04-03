import tensorflow as tf
import numpy as np
from itkutilities import get_itk_array
import utility


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 5-D tensor: [batch_size, width, height, depth, channels]
    # Our input images are 256x256x256 pixels, and have one color channel
    input_layer = tf.reshape(features["x"], [-1, 64, 64, 1])

    conv1 = tf.layers.conv2d(inputs=input_layer, filters=10, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)

    conv2 = tf.layers.conv2d(inputs=conv1, filters=20, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

    conv3 = tf.layers.conv2d(inputs=conv2, filters=30, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

    conv4 = tf.layers.conv2d(inputs=conv3, filters=40, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

    conv5 = tf.layers.conv2d(inputs=conv4, filters=40, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

    conv6 = tf.layers.conv2d(inputs=conv5, filters=30, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

    conv7 = tf.layers.conv2d(inputs=conv6, filters=20, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

    conv8 = tf.layers.conv2d(inputs=conv7, filters=10, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

    conv9 = tf.layers.conv2d(inputs=conv8, filters=5, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)

    dropout = tf.layers.dropout(inputs=conv9, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.conv2d(
        inputs=dropout,
        filters=2,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    print("Logits Shape: ", np.shape(logits))

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=3),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    epsilon = tf.constant(0.000001)

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

    part1_1 = tf.divide(tf.reduce_sum(log_probs_pos, name='reduce_pos_log'), tf.add(tf.to_float(num_pos), epsilon))
    part1_2 = tf.divide(tf.reduce_sum(log_probs_neg, name='reduce_neg_log'), tf.add(tf.to_float(num_neg), epsilon))
    loss = tf.add(tf.multiply(part1_1, -1), tf.multiply(part1_2, -1))

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
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


def load_train_data(filename, type=0):
    """Loads training and/or evaluation data"""
    print("Loading training data...")

    # 4 dimensional data
    # type == 0 is data, 1 is labeled
    if type == 0:
        data = np.asarray(get_itk_array(filename), dtype='float32')
    else:
        data = np.asarray(get_itk_array(filename), dtype='int32')

    print("All data loaded.")
    return data


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


def load_test_data(filename):
    """Loads testing data"""
    print("Started loading test data.")

    data = np.asarray(get_itk_array(filename), dtype='float32')

    print("Loaded test data.")

    return data


def load_test_data_folder(folder_name):

    # Read info file.
    with open(folder_name + "info.txt") as f:
        content = f.readlines()

    content = [x.strip() for x in content]
    num_files = int(content[4])

    print("Total Number of Files: ", num_files)
    print("Loading testing data...")

    data = np.zeros([num_files, 64, 64, 64], dtype='float32')

    for i in utility.my_range(0, num_files, 1):

        data_file_name = folder_name + "cropped" + str(i) + ".nii.gz"

        data[i] = np.asarray(get_itk_array(data_file_name), dtype='float32')

    print("All data loaded.")

    return data
