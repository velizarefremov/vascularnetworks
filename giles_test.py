import os
import sys
import numpy as np
from itkutilities import get_itk_array, write_itk_imageArray
import utility


if len(sys.argv) != 5:
    print("Usage: " + sys.argv[0] + " <testExecutable> <modelFile> <testData> <workFolder>")
    sys.exit(1)

testfilename = sys.argv[1]
modelfilename = sys.argv[2]
datafilename = sys.argv[3]
workfolder = sys.argv[4]

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

# Break here the file into smaller chunks.
index = 0

segment = False
if segment:
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
                # Apply segmentation to it.
                os.system("THEANO_FLAGS='device=gpu,floatX=float32' python " + testfilename +
                          " --model " + modelfilename + " " + workfolder + "/cropped" + str(index) + ".nii.gz")

                # Update indices
                startZ = startZ + stepsize
                index = index + 1

            startY = startY + stepsize
            startZ = 0

        startX = startX + stepsize
        startY = 0

# Combine the pieces back together to form the result

index = 0
startX = 0
startY = 0
startZ = 0

outputimg = np.zeros(inputSize)
confimg = np.zeros(inputSize)

for i in utility.my_range(0, xSize, stepsize):

    for j in utility.my_range(0, ySize, stepsize):

        for k in utility.my_range(0, zSize, stepsize):
            print "Step at: (", index, ")", startX, startY, startZ
            endX = startX + stepsize
            endY = startY + stepsize
            endZ = startZ + stepsize

            outputimg[startX:endX, startY:endY, startZ:endZ] = get_itk_array(workfolder +
                                                                             "/cropped" + str(index) + "_bins.nii.gz")

            confimg[startX:endX, startY:endY, startZ:endZ] = get_itk_array(workfolder +
                                                                             "/cropped" + str(index) + "_probs.nii.gz")
            # Update indices
            startZ = startZ + stepsize
            index = index + 1

        startY = startY + stepsize
        startZ = 0

    startX = startX + stepsize
    startY = 0

write_itk_imageArray(outputimg, workfolder + "/result_bin.nii.gz")
write_itk_imageArray(confimg, workfolder + "/result_conf.nii.gz")

