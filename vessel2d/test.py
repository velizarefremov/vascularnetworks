from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import utility
import tensorflow as tf
from vessel2d.model import cnn_model_fn, load_test_data
from itkutilities import write_itk_imageArray


def main(unused_args):

    test_data = load_test_data('../output/cropped47.nii.gz', )
    # print("Input Data...")
    # print("Shape: ", np.shape(test_data))
    # print("Max: ", np.max(test_data))
    # print("Min: ", np.min(test_data))
    # print("Mean: ", np.mean(test_data))

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/home/joseph/Desktop/models/giles2d")

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_data},
        batch_size=1,
        num_epochs=1,
        shuffle=False)

    # Predict data
    predictions = mnist_classifier.predict(input_fn=predict_input_fn)

    predictions = np.array(list(predictions))

    print(np.shape(predictions))
    print(predictions[0]['classes'])

    result = np.zeros([64, 64, 64], dtype='uint8')

    for i in utility.my_range(0, 64, 1):
        result[i] = predictions[i]['classes']

    # print(result)
    # print(np.shape(result))

    write_itk_imageArray(result, '../output/detected.nii.gz')

    # print("Predictions...")
    # print("Shape: ", np.shape(predictions))
    # print("Max: ", np.max(predictions))
    # print("Min: ", np.min(predictions))


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
