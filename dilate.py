#!/usr/bin/env python

import sys
import itk
itk.auto_progress(2)

if len(sys.argv) != 4:
    print("Usage: " + sys.argv[0] + " <inputImage> <outputImage> <radius>")
    sys.exit(1)

inputImage = sys.argv[1]
outputImage = sys.argv[2]
radiusValue = int(sys.argv[3])

PixelType = itk.UC
Dimension = 3

ImageType = itk.Image[PixelType, Dimension]

reader = itk.ImageFileReader[ImageType].New()
reader.SetFileName(inputImage)

StructuringElementType = itk.FlatStructuringElement[Dimension]
structuringElement = StructuringElementType.Ball(radiusValue)

grayscaleFilter = itk.GrayscaleDilateImageFilter[
    ImageType, ImageType, StructuringElementType].New()
grayscaleFilter.SetInput(reader.GetOutput())
grayscaleFilter.SetKernel(structuringElement)

writer = itk.ImageFileWriter[ImageType].New()
writer.SetFileName(outputImage)
writer.SetInput(grayscaleFilter.GetOutput())

writer.Update()
