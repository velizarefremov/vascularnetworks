import os
import sys
import numpy as np
from itkutilities import get_itk_array, write_itk_imageArray
import utility
from skimage.morphology import skeletonize_3d
import matplotlib.pyplot as plt

display = False

if len(sys.argv) != 3:
    print("Usage: " + sys.argv[0] + " <testData> <output>")
    sys.exit(1)

inputfile = sys.argv[1]
outputfile = sys.argv[2]

inputimg = np.array(get_itk_array(inputfile), dtype="uint8")

# perform skeletonization
skeleton = skeletonize_3d(inputimg)

if display == True:
    # display results
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                            sharex=True, sharey=True,
                            subplot_kw={'adjustable': 'box-forced'})

    ax = axes.ravel()

    ax[0].imshow(inputimg[55], cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('original', fontsize=20)

    ax[1].imshow(skeleton[55], cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[1].set_title('skeleton', fontsize=20)

    fig.tight_layout()
    plt.show()

write_itk_imageArray(skeleton, outputfile)
