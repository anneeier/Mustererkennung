import cv2 
import numpy as np
from skimage.metrics import structural_similarity as ssim

#Ähnlichkeit von Bildern: SSIM

def contrast(img, inputAlpha):
    new_img = np.zeros(img.shape, img.dtype)

    alpha = inputAlpha  #Einfache Konstraststeuerung
    beta = 0 # Einfache Helligkeitssteuerung

    #Ausführen der Anpassung
    for y in range(img.shape[0]):
        for x in range (img.shape[1]):
            new_img[y,x] = np.clip(alpha*img[y,x], 0, 255)
            1.935
    return new_img

def luminance(img, inputBeta):
    new_img = np.zeros(img.shape, img.dtype)

    alpha = 1  #Einfache Konstraststeuerung
    beta = inputBeta # Einfache Helligkeitssteuerung

    #Ausführen der Anpassung
    for y in range(img.shape[0]):
        for x in range (img.shape[1]):
            new_img[y,x] = np.clip(alpha*img[y,x] + beta, 0, 255)
            
    return new_img

def blur(img):
    blurImg = cv2.blur(img, (10,10))
    return blurImg


img = cv2.imread('Utils/LennaCol.png')
imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
bright_contrast_img = contrast(imgGray, 1.935)
dark_contrast_img = contrast(imgGray, 0.485)
bright_luminance_img = luminance(imgGray, 115)
dark_luminance_img = luminance(imgGray, -51.5)
blur_img = blur(imgGray)

cv2.imshow('Originial', imgGray)
cv2.imshow('Bright Contrast', bright_contrast_img)
cv2.imshow('Dark Contrast', dark_contrast_img)
cv2.imshow('Blur', blur_img)
cv2.imshow('Bright Luminance', bright_luminance_img)
cv2.imshow('Dark Luminance', dark_luminance_img)


result = ssim(imgGray, bright_contrast_img)
result1 = ssim(imgGray, dark_contrast_img)
result2 = ssim(imgGray, blur_img)
result3 = ssim(imgGray, bright_luminance_img)
result4 = ssim(imgGray, dark_luminance_img)

print('Result Bright Contrast:', result)
print('Result Dark Contrast:', result1)
print('Result Blur:', result2)
print('Result Bright Luminance:' , result3)
print('Result Dark Luminance:' , result4)

cv2.waitKey()

#Quellen:
# https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html
# https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
# https://www.geeksforgeeks.org/opencv-python-program-to-blur-an-image/



#Steganographie: Binärbild in Blau-Kanal

grayImg = cv2.imread('Utils/th.jpg', cv2.IMREAD_GRAYSCALE)

bitImages = []
for bit in range(8):
    bitMask = 1 << bit
    bitPlane = cv2.bitwise_and(grayImg, bitMask)
    bitImages.append(bitPlane)

#for i, bitImages in enumerate(bitImages):
#    cv2.imwrite(f'bit_{i}.png', bitImages)
