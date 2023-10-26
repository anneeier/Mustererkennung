import cv2 
import numpy as np
from PIL import Image
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

grayImg = cv2.imread('Utils/th.jpg', cv2.IMREAD_GRAYSCALE)

bitImages = []
for bit in range(8):
    bitMask = 1 << bit
    bitPlane = cv2.bitwise_and(grayImg, bitMask)
    bitImages.append(bitPlane)

for i, bitImages in enumerate(bitImages):
    cv2.imwrite(f'Utils/bit_{i}.jpg', bitImages)

# Convert cover image to gray-scale
cover = Image.open("Utils/th.jpg").convert('L')

data_c = np.array(cover)

# Convert image to 1-bit pixel, black and white and resize to cover image
secret = Image.open("Utils/bit_0Code.jpg").convert('1')


data_s = np.array(secret, dtype=np.uint8)

# Rewrite LSB
res = data_c & ~1 | data_s

new_img = Image.fromarray(res).convert("L")
new_img.save("Utils/cover-secret.jpg")

secret_img = cv2.imread("Utils/cover-secret.jpg", cv2.IMREAD_GRAYSCALE)

cv2.imshow("Test", grayImg)
cv2.imshow("Code", secret_img)

result = ssim(grayImg, secret_img)

print('Result Bright Contrast:', result)

cv2.waitKey()



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

