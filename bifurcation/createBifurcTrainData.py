import os
import sys
import numpy as np
from itkutilities import get_itk_array, write_itk_imageArray
import utility

if len(sys.argv) != 5:
    print("Usage: " + sys.argv[0] + " <rawData> <bifPointsData> <workFolder> <blockSize>")
    sys.exit(1)

datafilename = sys.argv[1]
biffilename = sys.argv[2]
workfolder = sys.argv[3]
blocksize = int(sys.argv[4])

inputimg = get_itk_array(datafilename)
bifimg = get_itk_array(biffilename)

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

for i in utility.my_range(blocksize, xSize - blocksize, 1):

    print("Step", i)

    for j in utility.my_range(blocksize, ySize - blocksize, 1):

        for k in utility.my_range(blocksize, zSize - blocksize, 1):
            # print("Step at: (", index, ")", i, j, k)

            # If bifurcation exists.
            if bifimg[i, j, k] != 0:
                # Crop image and write it.
                cropped = inputimg[(i - blocksize):(i + blocksize + 1), (j - blocksize):(j + blocksize + 1),
                          (k - blocksize):(k + blocksize + 1)]

                write_itk_imageArray(cropped, workfolder + "cropped" + str(index) + ".nii.gz")

                # Update indices
                index = index + 1

print("Total bifurcations: ", index)

file.write(str(index) + "\n")
file.close()

print("Done.")
