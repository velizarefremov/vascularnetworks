from itkutilities import get_itk_array, write_itk_imageArray
import numpy as np
import sys

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Usage: " + sys.argv[0] + " <inputImage> <outputImage>")
        sys.exit(1)

    inputimg = np.array(get_itk_array(sys.argv[1]), dtype="uint8")
    outputimg = sys.argv[2]

    inputimg = np.subtract(inputimg, inputimg.min())

    inputimg = np.multiply(inputimg, 1.0 / inputimg.max(), casting='unsafe')

    write_itk_imageArray(np.asarray(inputimg), outputimg)
