import os
import sys
import numpy as np
import utility


if len(sys.argv) != 4:
    print("Usage: " + sys.argv[0] + " <normalizeexe> <splitFolder> <normalFolder>")
    sys.exit(1)

executable = sys.argv[1]
workfolder = sys.argv[2]
normfolder = sys.argv[3]

# Copy info file also here.
os.system("cp " + workfolder + "/info.txt" + " " + normfolder + "/info.txt")

# Read info file.
with open(normfolder + "/info.txt") as f:
    content = f.readlines()

content = [x.strip() for x in content]

stepsize = content[0]
xSize = content[1]
ySize = content[2]
zSize = content[3]
numfiles = content[4]

print("Step Size is: ", stepsize)
print("Total Image Size is: ", xSize, ySize, zSize)
print("Total Number of Files: ", numfiles)

# Segment every file.
index = 0

for i in utility.my_range(0, int(numfiles), 1):

    print("Segmenting file: ", "cropped" + str(index) + ".nii.gz")

    datafilename = workfolder + "/cropped" + str(index) + ".nii.gz"
    writefilename = normfolder + "/cropped" + str(index) + ".nii.gz"

    # Apply thresholding to it.
    os.system("python " + executable + " " + datafilename + " " + writefilename)

    # Update indices
    index = index + 1

print("Done.")
