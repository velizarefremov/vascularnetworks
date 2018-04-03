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

    # Convolutional Layer #1
    # Computes 5 features using a 3x3x3 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 256, 256, 256, 1]
    # Output Tensor Shape: [batch_size, 256, 256, 256, 5]
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=5, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)

    # Convolutional Layer #2
    # Computes 10 features using a 5x5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 256, 256, 256, 5]
    # Output Tensor Shape: [batch_size, 256, 256, 256, 10]
    conv2 = tf.layers.conv2d(inputs=conv1, filters=10, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

    # Convolutional Layer #3
    # Computes 20 features using a 5x5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 256, 256, 256, 10]
    # Output Tensor Shape: [batch_size, 256, 256, 256, 20]
    conv3 = tf.layers.conv2d(inputs=conv2, filters=20, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

    # Convolutional Layer #4
    # Computes 50 features using a 3x3x3 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 256, 256, 256, 20]
    # Output Tensor Shape: [batch_size, 256, 256, 256, 50]
    conv4 = tf.layers.conv2d(inputs=conv3, filters=50, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)


    # Convolutional Layer #5
    # Computes 2 features using a 3x3x3 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 256, 256, 256, 50]
    # Output Tensor Shape: [batch_size, 256, 256, 256, 2]
    logits = tf.layers.conv2d(inputs=conv4, filters=2, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)

    # print("Logits Shape: ", np.shape(logits))
    # print("Labels Shape: ", np.shape(labels))

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=3),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # print("Preds: ", predictions['probabilities'][:, :, :, 1])

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
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


def load_test_data_folder(folder_name, number_files=0):

    # Read info file.
    with open(folder_name + "info.txt") as f:
        content = f.readlines()

    content = [x.strip() for x in content]
    num_files = int(content[4])

    print("Total Number of Files: ", num_files)
    print("Loading testing data...")

    if number_files != 0:
        num_files = number_files
        print("Loading only the first ", num_files, " of data.")

    data = np.zeros([num_files, 64, 64, 64], dtype='float32')

    for i in utility.my_range(0, num_files, 1):

        data_file_name = folder_name + "cropped" + str(i) + ".nii.gz"

        data[i] = np.asarray(get_itk_array(data_file_name), dtype='float32')

    print("All data loaded.")

    return data

    # Calculate Loss (for both TRAIN and EVAL modes)
    # epsilon = tf.constant()
    #
    # num_pos = tf.reduce_sum(labels, name='label_pos')
    # num_neg = tf.subtract(204800, tf.reduce_sum(labels), name='label_neg')
    #
    # flat_probs_pos = tf.reshape(predictions['probabilities'][:, :, :, 1], [-1])
    # flat_probs_neg = tf.reshape(predictions['probabilities'][:, :, :, 0], [-1])
    # flat_labels_pos = tf.reshape(labels, [-1])
    # flat_labels_neg = tf.add(tf.multiply(flat_labels_pos, -1), 1)
    #
    # pos_log_input = tf.add(tf.multiply(tf.to_float(flat_labels_pos), flat_probs_pos, name='mult_pos'),
    #                        tf.to_float(flat_labels_neg))
    # neg_log_input = tf.add(tf.multiply(tf.to_float(flat_labels_neg), flat_probs_neg, name='mult_neg'),
    #                        tf.to_float(flat_labels_pos))
    #
    # log_probs_pos = tf.log(pos_log_input, name='log_pos')
    # log_probs_neg = tf.log(neg_log_input, name='log_neg')
    #
    # part1_1 = tf.divide(tf.reduce_sum(log_probs_pos, name='reduce_pos_log'), tf.to_float(tf.add(num_pos, epsilon)))
    # part1_2 = tf.divide(tf.reduce_sum(log_probs_neg, name='reduce_neg_log'), tf.to_float(num_neg))
    # loss1 = tf.add(tf.multiply(part1_1, -1), tf.multiply(part1_2, -1))
    #
    # # LOSS Part 2
    # flat_preds_pos = tf.to_int32(tf.reshape(predictions['classes'][:, :, :], [-1]))
    # flat_preds_neg = tf.add(tf.multiply(flat_preds_pos, -1), 1)
    #
    # false_pos_preds = tf.multiply(flat_preds_pos, flat_labels_neg, name='false_pos')
    # false_neg_preds = tf.multiply(flat_preds_neg, flat_labels_pos, name='false_neg')
    #
    # num_false_pos = tf.to_float(tf.reduce_sum(false_pos_preds), name='num_false_pos')
    # num_false_neg = tf.to_float(tf.reduce_sum(false_neg_preds), name='num_false_neg')
    #
    # false_pos_preds_neg = tf.multiply(tf.add(tf.multiply(tf.to_float(false_pos_preds), -1), 1), 0.5)
    # false_neg_preds_neg = tf.multiply(tf.add(tf.multiply(tf.to_float(false_neg_preds), -1), 1), 0.5)
    #
    # false_pos_input = tf.add(tf.multiply(tf.to_float(false_pos_preds), flat_probs_neg, name='false_pos_in')
    #                          , false_pos_preds_neg)
    # false_neg_input = tf.add(tf.multiply(tf.to_float(false_neg_preds), flat_probs_pos, name='false_neg_in')
    #                          , false_neg_preds_neg)
    #
    # gamma1 = tf.add(tf.divide(tf.reduce_sum(tf.subtract(false_pos_input, 0.5)), num_false_pos), 0.5)
    # gamma2 = tf.add(tf.divide(tf.reduce_sum(tf.subtract(false_neg_input, 0.5)), num_false_neg), 0.5)
    #
    # false_pos_preds_neg_2 = tf.add(tf.multiply(false_pos_preds, -1), 1)
    # false_neg_preds_neg_2 = tf.add(tf.multiply(false_neg_preds, -1), 1)
    #
    # false_pos_log_input = tf.add(tf.multiply(tf.to_float(false_pos_preds), tf.to_float(flat_probs_neg),
    #                                          name='mult_pos'), tf.to_float(false_pos_preds_neg_2))
    # false_neg_log_input = tf.add(tf.multiply(tf.to_float(false_neg_preds), tf.to_float(flat_preds_pos),
    #                                          name='mult_neg'), tf.to_float(false_neg_preds_neg_2))
    #
    # part2_1 = tf.divide(tf.reduce_sum(false_pos_log_input, name='reduce_pos_log_p2'), tf.to_float(num_pos))
    # part2_2 = tf.divide(tf.reduce_sum(false_neg_log_input, name='reduce_neg_log_p2'), tf.to_float(num_neg))
    # loss2 = tf.add(tf.multiply(part2_1, -gamma1), tf.multiply(part2_2, -gamma2))
    #
    # loss = tf.add(loss1, loss2)

# LOSS SIMPLE
#     num_pos = tf.reduce_sum(labels, name='label_pos')
#     pos_ratio = tf.divide(num_pos, 204800)
#
#     crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
#     weights = tf.add(tf.multiply(tf.to_float(labels), (1.0-tf.to_float(pos_ratio))), tf.to_float(pos_ratio))
#     loss = tf.reduce_mean(tf.multiply(crossent, weights))
