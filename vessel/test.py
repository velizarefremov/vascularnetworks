from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from vessel.model import cnn_model_fn, load_train_data_folder
from itkutilities import write_itk_imageArray


def main(unused_args):

    test_data = load_train_data_folder('../input/norm1/', data_type=0, number_files=4)
    test_data = np.reshape(test_data, [-1, 64, 64, 64])
    # print("Input Data...")
    # print("Shape: ", np.shape(test_data))
    # print("Max: ", np.max(test_data))
    # print("Min: ", np.min(test_data))
    # print("Mean: ", np.mean(test_data))

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/home/joseph/Desktop/models/gilestest")

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_data},
        batch_size=1,
        num_epochs=1,
        shuffle=False)

    # Predict data
    predictions = mnist_classifier.predict(input_fn=predict_input_fn)

    predictions = np.array(list(predictions))

    print(predictions[0])

    result = predictions[0]['classes']

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
