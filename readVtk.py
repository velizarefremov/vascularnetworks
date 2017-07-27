from vtk import *
import numpy as np
from matplotlib import pyplot as plt

file_name = "av_cropped.vtk"

reader = vtkUnstructuredGridReader()
reader.SetFileName(file_name)
reader.ReadAllScalarsOn()
reader.Update()
output = reader.GetOutput()

num_points = output.GetNumberOfPoints()
num_cells = output.GetNumberOfCells()

# print num_points
# print num_cells

# print output.GetPoint(2)[2]
# print output.GetPoint(3)

# print output.GetCell(0).GetNumberOfPoints()
# print output.GetCell(0).GetPointIds().GetId(0)
# print output.GetCell(0).GetPointIds().GetId(1)

celldata = output.GetCellData().GetArray('str_radiusAdjusted')

# print celldata.GetSize()

radius = np.zeros(celldata.GetSize())
xval = np.zeros(num_points)
yval = np.zeros(num_points)
zval = np.zeros(num_points)
idsStart = np.zeros(num_cells)
idsEnd = np.zeros(num_cells)

# Fill X,Y,Z Values into Numpy Array
for i in range(0, num_points):
    data = output.GetPoint(i)
    xval[i] = data[0]
    yval[i] = data[1]
    zval[i] = data[2]

for i in range(0, num_cells):
    data = output.GetCell(i).GetPointIds()
    idsStart[i] = data.GetId(0)
    idsEnd[i] = data.GetId(1)

# Fill Diameters into Numpy Array
for i in range(0, num_cells):
    radius[i] = celldata.GetValue(i)

print min(radius)
print max(radius)
print np.mean(radius)
print np.std(radius)

print min(xval), max(xval)
print min(yval), max(yval)
print min(zval), max(zval)
