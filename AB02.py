import cv2 
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

#Ähnlichkeit von Bildern: SSIM

def contrast(img, inputAlpha):
    new_img = np.zeros(img.shape, img.dtype)

    alpha = inputAlpha  #Einfache Konstraststeuerung

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


# img = cv2.imread('Utils/LennaCol.png')
# imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# bright_contrast_img = contrast(imgGray, 1.935)
# dark_contrast_img = contrast(imgGray, 0.485)
# bright_luminance_img = luminance(imgGray, 115)
# dark_luminance_img = luminance(imgGray, -51.5)
# blur_img = blur(imgGray)

# cv2.imshow('Originial', imgGray)
# cv2.imshow('Bright Contrast', bright_contrast_img)
# cv2.imshow('Dark Contrast', dark_contrast_img)
# cv2.imshow('Blur', blur_img)
# cv2.imshow('Bright Luminance', bright_luminance_img)
# cv2.imshow('Dark Luminance', dark_luminance_img)


# result = ssim(imgGray, bright_contrast_img)
# result1 = ssim(imgGray, dark_contrast_img)
# result2 = ssim(imgGray, blur_img)
# result3 = ssim(imgGray, bright_luminance_img)
# result4 = ssim(imgGray, dark_luminance_img)

# print('Result Bright Contrast:', result)
# print('Result Dark Contrast:', result1)
# print('Result Blur:', result2)
# print('Result Bright Luminance:' , result3)
# print('Result Dark Luminance:' , result4)

# cv2.waitKey()

#Quellen:
# https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html
# https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
# https://www.geeksforgeeks.org/opencv-python-program-to-blur-an-image/



#Steganographie: Binärbild in Blau-Kanal

import cv2
import numpy as np

# Lade das Farbbild, in das du das andere Bild einbetten möchtest
host_image = cv2.imread('Utils/th.jpg')

# Lade das Bild, das du einbetten möchtest
embedded_image = cv2.imread('Utils/bit_0Code.jpg')

# Überprüfe, ob die eingebettete Bilddatei in das Hostbild passt
if embedded_image.shape[0] > host_image.shape[0] or embedded_image.shape[1] > host_image.shape[1]:
    raise ValueError("Das eingebettete Bild ist zu groß, um in das Hostbild einzubetten.")

# Iteriere durch die Pixel des eingebetteten Bildes und setze die LSBs der Pixel des Hostbildes
for row in range(embedded_image.shape[0]):
    for col in range(embedded_image.shape[1]):
        for channel in range(embedded_image.shape[2]):
            host_pixel = host_image[row, col, channel]
            embedded_pixel = embedded_image[row, col, channel]

            # Lösche die letzten beiden Bits des Hostpixels und füge die beiden LSBs des eingebetteten Pixels hinzu
            host_pixel = (host_pixel & 0b11111100) | (embedded_pixel >> 6)

            # Aktualisiere das Pixel im Hostbild
            host_image[row, col, channel] = host_pixel

# Speichere das modifizierte Hostbild mit dem eingebetteten Bild
cv2.imwrite('Utils/result_image.jpg', host_image)

final_img = cv2.imread('Utils/result_image.jpg', cv2.IMREAD_COLOR)
img = cv2.imread('Utils/th.jpg', cv2.IMREAD_COLOR)

cv2.imshow('Original', img)
cv2.imshow('Code', final_img)

# result = ssim(img, final_img)
# print('Result Bright Contrast:', result)

cv2.waitKey()

# img = cv2.imread('Utils/th.jpg')
# b, g, r = cv2.split(img)
# img_b = b

# bitImages = []
# for bit in range(8):
#     bitMask = 1 << bit
#     bitPlane = cv2.bitwise_and(img_b, bitMask)
#     bitImages.append(bitPlane)

# for i, bitImages in enumerate(bitImages):
#     cv2.imwrite(f'Utils/bit_{i}.jpg', bitImages)

# # Convert cover image to gray-scale
# cover = img_b
# data_c = np.array(cover)

# # Convert image to 1-bit pixel, black and white and resize to cover image
# secret = cv2.imread("Utils/bit_0Code.jpg")
# data_s = np.array(secret, dtype=np.uint8)

# # Rewrite LSB
# res = data_c & ~1 | data_s

# new_img = Image.fromarray(res)
# new_img.save("Utils/cover-secret1.jpg")
# rgb = cv2.imread("Utils/cover-secret1.jpg", cv2.IMREAD_COLOR)

# cv2.imshow("Original", img)
# cv2.imshow("Original + Code", rgb)
# cv2.imshow("r", r)
# #result = ssim(grayImg, secret_img)

# #print('Result Bright Contrast:', result)

# cv2.waitKey()

#Quellen:
# https://medium.com/@stephanie.werli/image-steganography-with-python-83381475da57

# lsbImg = cv2.imread('Utils/bit_0.png')

# new_img = np.ones(lsbImg.shape, lsbImg.dtype)

# #Ausführen der Anpassung
# for y in range(new_img.shape[0]):
#     for x in range (new_img.shape[1]):
#         new_img[y,x] = lsbImg[y,x] + grayImg[y,x]


            
# cv2.imshow('Test', new_img)
# cv2.imshow('Ori', grayImg)
# cv2.imshow('code', lsbImg)
# cv2.waitKey()

