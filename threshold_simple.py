from itkutilities import get_itk_array, write_itk_imageArray
import numpy as np
import sys

if __name__ == '__main__':

    if len(sys.argv) != 4:
        print("Usage: " + sys.argv[0] + " <inputImage> <outputImage> <threshold>")
        sys.exit(1)

    inputimg = get_itk_array(sys.argv[1])
    outputimg = sys.argv[2]
    threshold = float(sys.argv[3])

    write_itk_imageArray(np.asarray(inputimg >= threshold, dtype='uint8'), outputimg)
