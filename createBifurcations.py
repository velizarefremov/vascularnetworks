from vtk import *
import numpy as np
from matplotlib import pyplot as plt
import itkutilities
import sys

xwidth = 580
ywidth = 640
zwidth = 136

xscale = 0.3125
yscale = 0.3125
zscale = 0.6

input_file_name = sys.argv[1]
output_file_name = sys.argv[2]

reader = vtkPolyDataReader()
reader.SetFileName(input_file_name)
reader.ReadAllScalarsOn()
reader.Update()
output = reader.GetOutput()

num_points = output.GetNumberOfPoints()
num_cells = output.GetNumberOfCells()

print num_points
print num_cells

print output.GetPoint(2)[1]
print output.GetPoint(3)

print output.GetCell(0).GetNumberOfPoints()
print output.GetCell(0).GetPointIds().GetId(0)
print output.GetCell(0).GetPointIds().GetId(1)

celldata = output.GetCellData().GetArray('radius')

print celldata.GetSize()

radius = np.zeros(celldata.GetSize())

# Create image to write.
a = np.zeros(shape=(zwidth, ywidth, xwidth), dtype='uint8')

# Start and end IDs of edges
counts = np.zeros(num_points)

for i in range(0, num_cells):
    data = output.GetCell(i).GetPointIds()
    counts[data.GetId(0)] += 1
    counts[data.GetId(1)] += 1

# Indices start with 0.
for i in range(0, num_points):
    # If bifurcation
    if counts[i] > 2:
        # print "index: ", i, " with ", counts[i], " connections"
        pt = output.GetPoint(i)
        x = pt[0]/xscale
        y = pt[1]/yscale
        z = pt[2]/zscale
        for j in range(-2, 3):
            for k in range(-2, 3):
                for l in range(-2, 3):
                    xinc = x + j
                    yinc = y + k
                    zinc = z + l
                    if xinc < 0 or yinc < 0 or zinc < 0 or xinc >= xwidth or yinc >= ywidth or zinc >= zwidth:
                        continue
                    else:
                        a[zinc, yinc, xinc] = 1


# Write the image.
itkutilities.write_itk_imageArray(a, output_file_name)
