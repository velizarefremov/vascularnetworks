from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from vessel.model import cnn_model_fn, load_train_data


def main(unused_args):

    # Load train data
    train_data = load_train_data('..\\input\\vessel\\raw8.nii.gz', type=0)
    train_labels = load_train_data('..\\input\\vessel\\seg8.nii.gz', type=1)
    # train_labels = np.reshape(train_labels, [-1, 256, 256, 256, 1])

    eval_data = load_train_data('..\\input\\vessel\\raw8_2.nii.gz', type=0)
    eval_labels = load_train_data('..\\input\\vessel\\seg8_2.nii.gz', type=1)
    # eval_labels = np.reshape(eval_labels, [-1, 256, 256, 256, 1])

    print(np.shape(train_data))
    print(np.shape(train_labels))
    print(np.shape(eval_data))
    print(np.shape(eval_labels))

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/gilestest")

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
        steps=5000,
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
