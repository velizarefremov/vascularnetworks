from itkutilities import get_itk_array, write_itk_imageArray
import numpy as np
import sys

if __name__ == '__main__':

    if len(sys.argv) != 9:
        print("Usage: " + sys.argv[0] + " <inputImage> <outputImage> <startX> <startY> <startZ> <sizeX> <sizeY> <sizeZ>")
        sys.exit(1)

    inputimg = get_itk_array(sys.argv[1])
    outfile = sys.argv[2]

    startX = int(sys.argv[3])
    startY = int(sys.argv[4])
    startZ = int(sys.argv[5])
    sizeX = int(sys.argv[6])
    sizeY = int(sys.argv[7])
    sizeZ = int(sys.argv[8])

    endX = startX + sizeX
    endY = startY + sizeY
    endZ = startZ + sizeZ

    print(np.shape(inputimg))

    print(startX, startY, startZ)
    print(endX, endY, endZ)

    cropped = inputimg[startX:endX, startY:endY, startZ:endZ]

    write_itk_imageArray(cropped, outfile)
