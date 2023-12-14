import cv2 
import numpy as np
from PIL import Image

def readImagesAndTimes():
    filenames = [
        # "images/MEFDatabase/source image sequences/Balloons_Erik Reinhard/DSC_0163.png",
        # "images/MEFDatabase/source image sequences/Balloons_Erik Reinhard/DSC_0164.png",
        # "images/MEFDatabase/source image sequences/Balloons_Erik Reinhard/DSC_0165.png",
        "images/MEFDatabase/source image sequences/Balloons_Erik Reinhard/DSC_0166.png",
        "images/MEFDatabase/source image sequences/Balloons_Erik Reinhard/DSC_0167.png",
        "images/MEFDatabase/source image sequences/Balloons_Erik Reinhard/DSC_0168.png",
        "images/MEFDatabase/source image sequences/Balloons_Erik Reinhard/DSC_0169.png",
        "images/MEFDatabase/source image sequences/Balloons_Erik Reinhard/DSC_0170.png",
        "images/MEFDatabase/source image sequences/Balloons_Erik Reinhard/DSC_0171.png",
    ]

    images = []
    for filename in filenames:
        im = cv2.imread(filename)
        images.append(im)

    return images

images = readImagesAndTimes()

alignMTB = cv2.createAlignMTB()
alignMTB.process(images, images)

mergeMertens = cv2.createMergeMertens()
exposureFusion = mergeMertens.process(images)

cv2.imwrite('result3.png', exposureFusion *255)

cv2.imshow('Test',exposureFusion )
cv2.waitKey()