from itkutilities import get_itk_array, write_itk_imageArray
import numpy as np
import sys

if __name__ == '__main__':

    if len(sys.argv) != 5:
        print("Usage: " + sys.argv[0] + " <inputImage> <outputImage> <min> <max>")
        sys.exit(1)

    inputimg = get_itk_array(sys.argv[1])
    outputimg = sys.argv[2]
    min = int(sys.argv[3])
    max = int(sys.argv[4])

    inputimg.clip(min, max, out=inputimg)
    inputimg -= min
    multiplier = 255.0/float(max-min+1)
    print(multiplier)
    inputimg = np.multiply(inputimg, multiplier, out=inputimg, casting='unsafe')

    write_itk_imageArray(np.asarray(inputimg, dtype='uint8'), outputimg)
