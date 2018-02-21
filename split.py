import os
import sys
import numpy as np
from itkutilities import get_itk_array, write_itk_imageArray
import utility


if len(sys.argv) != 3:
    print("Usage: " + sys.argv[0] + " <testData> <workFolder>")
    sys.exit(1)

datafilename = sys.argv[1]
workfolder = sys.argv[2]

inputimg = get_itk_array(datafilename)

# File will be split into (stepsize x stepsize x stepsize) chunks
stepsize = 200

startX = 0
startY = 0
startZ = 0

inputSize = np.shape(inputimg)
xSize = inputSize[0]
ySize = inputSize[1]
zSize = inputSize[2]

file = open(workfolder + "info.txt", "w")
file.write(str(stepsize) + "\n")
file.write(str(xSize) + "\n")
file.write(str(ySize) + "\n")
file.write(str(zSize) + "\n")

# Break here the file into smaller chunks.
index = 0

for i in utility.my_range(0, xSize, stepsize):
    for j in utility.my_range(0, ySize, stepsize):

        for k in utility.my_range(0, zSize, stepsize):
            print "Step at: (", index, ")", startX, startY, startZ
            endX = startX + stepsize
            endY = startY + stepsize
            endZ = startZ + stepsize

            # Crop image and write it.
            cropped = inputimg[startX:endX, startY:endY, startZ:endZ]
            write_itk_imageArray(cropped, workfolder + "/cropped" + str(index) + ".nii.gz")

            # Update indices
            startZ = startZ + stepsize
            index = index + 1

        startY = startY + stepsize
        startZ = 0

    startX = startX + stepsize
    startY = 0

file.write(str(index) + "\n")
file.close()

print "Done."