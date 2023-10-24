import cv2 
import numpy as np
from skimage.metrics import structural_similarity as ssim

#Ähnlichkeit von Bildern: SSIM

img = cv2.imread('Utils/LennaCol.png')
imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

result = ssim(imgGray, imgGray)

print(result)


#Steganographie: Binärbild in Blau-Kanal

grayImg = cv2.imread('Utils/th.jpg', cv2.IMREAD_GRAYSCALE)

bitImages = []
for bit in range(8):
    bitMask = 1 << bit
    bitPlane = cv2.bitwise_and(grayImg, bitMask)
    bitImages.append(bitPlane)

for i, bitImages in enumerate(bitImages):
    cv2.imwrite(f'bit_{i}.png', bitImages)
