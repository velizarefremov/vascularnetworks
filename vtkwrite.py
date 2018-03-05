import vtk
import os
import sys
import numpy as np
from itkutilities import get_itk_array, write_itk_imageArray
import utility

if len(sys.argv) != 3:
    print("Usage: " + sys.argv[0] + " <input> <output>")
    sys.exit(1)

inputfile = sys.argv[1]
outputfile = sys.argv[2]

inputimage = np.array(get_itk_array(inputfile), dtype="uint8")

# print(np.shape(inputimage))

dims = np.shape(inputimage)

inputimg = np.zeros(dims + np.full(np.shape(dims), 2))

inputimg[1:dims[0]+1, 1:dims[1]+1, 1:dims[2]+1] = inputimage

# print(inputimg)

dims = np.shape(inputimg)

ptindex = 0

Points = vtk.vtkPoints()
line = vtk.vtkCellArray()
radius = vtk.vtkIntArray()
radius.SetNumberOfComponents(1)
radius.SetName("radius")

# line.Allocate(8)

print("started")

data = dict()

for i in utility.my_range(1, dims[0] - 1, 1):
    print("Step: ", i)

    for j in utility.my_range(1, dims[1] - 1, 1):

        for k in utility.my_range(1, dims[2] - 1, 1):

            # If pixel is 1
            if inputimg[i, j, k] == 1:

                # Get index at image
                ind = i * dims[0] * dims[1] + j * dims[1] + k

                # Check if already added to list. If not, add.
                if ind not in data:
                    Points.InsertNextPoint(i, j, k)
                    data[ind] = ptindex
                    mainind = ptindex
                    ptindex = ptindex + 1
                else:
                    mainind = data[ind]

                print("Point with index: ", ind, " : ", mainind, " at: ", i, j, k)

                # Now check neighbours around and add them.

                for ix in utility.my_range(i - 1, i + 2, 1):
                    for jx in utility.my_range(j - 1, j + 2, 1):
                        for kx in utility.my_range(k - 1, k + 2, 1):

                            if not (ix == i and jx == j and kx == k) and inputimg[ix, jx, kx] == 1:
                                # Add Point if not already added
                                ind = ix * dims[0] * dims[1] + jx * dims[1] + kx
                                if ind not in data:
                                    Points.InsertNextPoint(ix, jx, kx)
                                    data[ind] = ptindex
                                    secondindex = ptindex
                                    ptindex = ptindex + 1
                                else:
                                    secondindex = data[ind]

                                # Insert Line
                                line.InsertNextCell(2)
                                line.InsertCellPoint(mainind)
                                line.InsertCellPoint(secondindex)
                                radius.InsertNextValue(5)

print("Passed through the file")


# Add the vertices and edges to unstructured Grid
G = vtk.vtkPolyData()
G.SetPoints(Points)
G.SetLines(line)
G.GetCellData().AddArray(radius)

print("Writing to file: ", outputfile)

# Dump the graph in VTK unstructured format (.vtu)
gw = vtk.vtkPolyDataWriter()
gw.SetFileName(outputfile)
gw.SetInputData(G)
gw.Write()
