from itk_utilities import get_itk_array, write_itk_imageArray
import numpy as np
import sys

if __name__ == '__main__':

    if len(sys.argv) != 4:
        print("Usage: " + sys.argv[0] + " <inputImage> <maskImage> <outputImage>")
        # python apply_mask.py /home/joseph/Desktop/dataset_giles/raw/01.nii.gz /home/joseph/Desktop/dataset_giles/mask_dilated/01.nii.gz /home/joseph/Desktop/masked_dilated.nii.gz
        sys.exit(1)

    inputimg = get_itk_array(sys.argv[1])
    mask = get_itk_array(sys.argv[2])

    masked = np.multiply(inputimg, mask)

    write_itk_imageArray(masked, sys.argv[3])
