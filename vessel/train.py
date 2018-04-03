from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from vessel.model import cnn_model_fn, load_train_data_folder


def main(unused_args):

    train_data = load_train_data_folder('../input/norm1/', data_type=0, number_files=4)
    train_data = np.reshape(train_data, [-1, 64, 64, 64])
    train_labels = load_train_data_folder('../input/norm2/', data_type=1, number_files=4)
    train_labels = np.reshape(train_labels, [-1, 64, 64, 64])

    eval_data = load_train_data_folder('../input/seg1/', data_type=0, number_files=4)
    eval_data = np.reshape(eval_data, [-1, 64, 64, 64])
    eval_labels = load_train_data_folder('../input/seg2/', data_type=1, number_files=4)
    eval_labels = np.reshape(eval_labels, [-1, 64, 64, 64])

    print(np.shape(train_data))
    print(np.shape(train_labels))
    print(np.shape(eval_data))
    print(np.shape(eval_labels))

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/home/joseph/Desktop/models/gilestest")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=1,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=10000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)

    print(eval_results)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
