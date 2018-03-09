from itkutilities import get_itk_array, write_itk_imageArray
import numpy as np
import sys

if __name__ == '__main__':

    if len(sys.argv) != 4:
        print("Usage: " + sys.argv[0] + " <bifurcationImage> <skeletonImage> <outputImage>")
        sys.exit(1)

    bifimg = get_itk_array(sys.argv[1])
    skelimg = get_itk_array(sys.argv[2])
    outputimg = sys.argv[3]

    bifimg = np.multiply(bifimg, 2)
    bifimg = np.maximum(skelimg, bifimg)

    write_itk_imageArray(np.asarray(bifimg, dtype='uint8'), outputimg)
