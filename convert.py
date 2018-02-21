from itkutilities import get_itk_array, write_itk_imageArray
import numpy as np
import sys

if __name__ == '__main__':

    inputimg = np.array(get_itk_array(sys.argv[1]), dtype='uint8')
    
    # Write to File
    write_itk_imageArray(inputimg, sys.argv[2])
