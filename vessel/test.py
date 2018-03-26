from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import utility
import tensorflow as tf
from vessel.model import cnn_model_fn, load_test_data
from itkutilities import write_itk_imageArray


def main(unused_args):
    # Load training and eval data
    # mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    # train_data = mnist.train.images  # Returns np.array (55000, 784)
    # train_labels = np.asarray(mnist.train.labels, dtype=np.int32)  # (55000, 1)

    test_data = load_test_data('input\\test.mhd')
    # print("Input Data...")
    # print("Shape: ", np.shape(test_data))
    # print("Max: ", np.max(test_data))
    # print("Min: ", np.min(test_data))
    # print("Mean: ", np.mean(test_data))

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_data},
        batch_size=100,
        num_epochs=1,
        shuffle=False)

    # Predict data
    predictions = mnist_classifier.predict(input_fn=predict_input_fn)

    predictions = np.array(list(predictions))

    result = np.zeros(np.shape(predictions))

    for index in utility.my_range(0, np.shape(predictions)[0], 1):
        result[index] = predictions[index]['classes']

    # print(result)

    write_itk_imageArray(np.reshape(result, [128, 128, 128]), 'output\\detected.nii.gz')

    # print("Predictions...")
    # print("Shape: ", np.shape(predictions))
    # print("Max: ", np.max(predictions))
    # print("Min: ", np.min(predictions))


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
