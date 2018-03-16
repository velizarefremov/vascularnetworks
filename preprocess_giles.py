from itkutilities import get_itk_array, write_itk_imageArray
import numpy as np
import sys

if __name__ == '__main__':

    if len(sys.argv) != 5:
        print("Usage: " + sys.argv[0] + " <inputImage> <outputImage> <clampMin> <clampMax>")
        sys.exit(1)

    inputimg = np.array(get_itk_array(sys.argv[1]), dtype="uint8")
    outputimg = sys.argv[2]
    min = int(sys.argv[3])
    max = int(sys.argv[4])

    # Clamp
    inputimg = np.asarray(np.clip(inputimg, min, max, out=inputimg), dtype='uint8')

    # Normalize
    inputimg = np.subtract(inputimg, inputimg.min())

    inputimg = np.multiply(inputimg, 1.0 / inputimg.max(), casting='unsafe')

    # Quad Projection
    inputimg = np.multiply(inputimg, inputimg)

    inputimg = np.multiply(inputimg, 255)

    # Write to output file
    write_itk_imageArray(np.asarray(inputimg, dtype='uint8'), outputimg)
