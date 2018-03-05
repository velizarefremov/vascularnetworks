from itkutilities import get_itk_array, write_itk_imageArray
import numpy as np
import sys

if __name__ == '__main__':

    if len(sys.argv) != 5:
        print("Usage: " + sys.argv[0] + " <inputImage> <outputImage> <min> <max>")
        sys.exit(1)

    inputimg = np.array(get_itk_array(sys.argv[1]), dtype="uint8")
    outputimg = sys.argv[2]
    min = int(sys.argv[3])
    max = int(sys.argv[4])

    inputimg = np.clip(inputimg, min, max, out=inputimg)

    write_itk_imageArray(np.asarray(inputimg, dtype='uint8'), outputimg)
