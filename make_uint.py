from itk_utilities import get_itk_array, write_itk_imageArray
import numpy as np
import sys

if __name__ == '__main__':

    # Convert to uint8
    inputimg = np.array(get_itk_array(sys.argv[1]), dtype='uint8')
    # inputimg2 = np.array(get_itk_array(sys.argv[2]), dtype='uint8')
    # inputimg3 = np.array(get_itk_array(sys.argv[3]), dtype='uint8')
    
    # Concatanate
    # concatimg = np.concatenate((inputimg, inputimg2, inputimg3), axis=0)

    # Write to File
    write_itk_imageArray(inputimg, sys.argv[2])
