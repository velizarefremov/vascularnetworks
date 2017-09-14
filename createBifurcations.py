from vtk import *
import numpy as np
from matplotlib import pyplot as plt
import itkutilities


file_name = "01.vtk"

reader = vtkPolyDataReader()
reader.SetFileName(file_name)
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
a = np.zeros(shape=(500, 300, 300), dtype='uint8')

# Indices start with 0.
# TODO: Apply boundary check
for i in range(num_points):
    pt = output.GetPoint(i)
    x = pt[0]
    y = pt[1]
    z = pt[2]
    for j in range(-2, 3):
        for k in range(-2, 3):
            for l in range(-2, 3):
                a[x + j, y + k, z + l] = 1




# Write the image.
itkutilities.write_itk_imageArray(a, 'deneme.nii.gz')
