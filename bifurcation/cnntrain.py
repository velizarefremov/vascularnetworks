from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from bifurcation.cnnmodel import cnn_model_fn, load_train_data


def main(unused_args):
    # Load training and eval data
    # mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    # train_data = mnist.train.images  # Returns np.array (55000, 784)
    # train_labels = np.asarray(mnist.train.labels, dtype=np.int32)  # (55000, 1)

    train_count = 42500
    train_data = load_train_data(count=train_count, start=0)
    train_labels = np.zeros([2 * train_count], dtype=np.int32)
    train_labels[train_count:] = 1

    eval_count = 7500
    eval_data = load_train_data(count=eval_count, start=42500)
    eval_labels = np.zeros([2 * eval_count], dtype=np.int32)
    eval_labels[eval_count:] = 1

    print(np.shape(train_data))
    print(np.shape(train_labels))
    print(np.shape(eval_data))
    print(np.shape(eval_labels))

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=50000,
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
