from itkutilities import get_itk_array, write_itk_imageArray
import numpy as np
import sys

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Usage: " + sys.argv[0] + " <inputImage> <outputImage>")
        sys.exit(1)

    inputimg = get_itk_array(sys.argv[1])
    outputimg = sys.argv[2]

    inputimg = np.multiply(inputimg, inputimg)

    inputimg = np.multiply(inputimg, 255)

    write_itk_imageArray(np.asarray(inputimg, dtype='uint8'), outputimg)
