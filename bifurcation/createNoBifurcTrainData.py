import os
import sys
import numpy as np
from itkutilities import get_itk_array, write_itk_imageArray
import utility

if len(sys.argv) != 7:
    print("Usage: " + sys.argv[0] + " <rawData> <bifPointsData> <segData> <workFolder> <blockSize> <stepSize>")
    sys.exit(1)

datafilename = sys.argv[1]
biffilename = sys.argv[2]
segfilename = sys.argv[3]
workfolder = sys.argv[4]
blocksize = int(sys.argv[5])
stepsize = int(sys.argv[6])

inputimg = get_itk_array(datafilename)
bifimg = get_itk_array(biffilename)
segimg = get_itk_array(segfilename)

inputSize = np.shape(bifimg)
xSize = inputSize[0]
ySize = inputSize[1]
zSize = inputSize[2]

file = open(workfolder + "info.txt", "w")
file.write(str(blocksize) + "\n")
file.write(str(xSize) + "\n")
file.write(str(ySize) + "\n")
file.write(str(zSize) + "\n")

print(blocksize, xSize, ySize, zSize)

# Index for bifurcations
index = 0

for i in utility.my_range(blocksize + stepsize, xSize - blocksize, 1):

    print("Step", i)

    for j in utility.my_range(blocksize + stepsize, ySize - blocksize, 1):

        for k in utility.my_range(blocksize + stepsize, zSize - blocksize, 1):
            # print("Step at: (", index, ")", i, j, k)

            # If bifurcation exists.
            if bifimg[i, j, k] != 0:
                # Crop image and write it.

                # Checks surrounding. If no bifurc. Add it.
                surrounding = bifimg[(i - 2 * stepsize - 1):i,
                              (j - 2 * stepsize):j,
                              (k - 2 * stepsize):k]

                # If no bifurc
                if np.max(surrounding) == 0:

                    cropped = inputimg[(i - blocksize - stepsize):(i + blocksize - stepsize + 1),
                              (j - blocksize - stepsize):(j + blocksize + 1 - stepsize),
                              (k - blocksize - stepsize):(k + blocksize + 1 - stepsize)]

                    # cropped_seg = segimg[(i - blocksize - stepsize):(i + blocksize + 1 - stepsize),
                    #              (j - blocksize - stepsize):(j + blocksize + 1 - stepsize),
                    #              (k - blocksize - stepsize):(k + blocksize + 1 - stepsize)]

                    write_itk_imageArray(cropped, workfolder + "cropped" + str(index) + ".nii.gz")
                    # write_itk_imageArray(cropped_seg, workfolder + "cropped" + str(index) + "_seg.nii.gz")

                    # Update indices
                    index = index + 1

print("Total bifurcations: ", index)

file.write(str(index) + "\n")
file.close()

print("Done.")
