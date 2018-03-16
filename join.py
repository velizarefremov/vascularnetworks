import os
import sys
import numpy as np
from itkutilities import get_itk_array, write_itk_imageArray
import utility


if len(sys.argv) != 3:
    print("Usage: " + sys.argv[0] + " <outputfile> <splitfolder>")
    sys.exit(1)

outputfile = sys.argv[1]
splitfolder = sys.argv[2]

# Read info file.
with open(splitfolder + "/info.txt") as f:
    content = f.readlines()

content = [x.strip() for x in content]

stepsize = int(content[0])
xSize = int(content[1])
ySize = int(content[2])
zSize = int(content[3])
numfiles = int(content[4])

startX = 0
startY = 0
startZ = 0

# Create a file with that size.
joinedfile = np.zeros([xSize, ySize, zSize], dtype="uint8")

# Join the small files into one big file.
index = 0

for i in utility.my_range(0, xSize, stepsize):
    for j in utility.my_range(0, ySize, stepsize):

        for k in utility.my_range(0, zSize, stepsize):
            print("Step at: (", index, ")", startX, startY, startZ)
            endX = startX + stepsize
            endY = startY + stepsize
            endZ = startZ + stepsize

            currentfilename = splitfolder + "/cropped" + str(index) + ".nii.gz"

            currentfile = np.array(get_itk_array(currentfilename), dtype='uint8')
            joinedfile[startX:endX, startY:endY, startZ:endZ] = currentfile

            # Update indices
            startZ = startZ + stepsize
            index = index + 1

        startY = startY + stepsize
        startZ = 0

    startX = startX + stepsize
    startY = 0


# Write the file on to disk.
print("Writing File to Disk...")
write_itk_imageArray(joinedfile, outputfile)
print("Done.")