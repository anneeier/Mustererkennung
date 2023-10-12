import cv2 as cv
import numpy as np

def sepia(img2sepia):
    imgGray = cv.cvtColor(imgSrc, cv.COLOR_RGB2GRAY)
    normalizedGary = np.array(imgGray, np.float32)/255
    sepia = np.ones(imgSrc.shape)
    sepia[:,:,0] *= 100 #B
    sepia[:,:,1] *= 200 #G
    sepia[:,:,2] *= 255 #R
    
    sepia[:,:,0] *= normalizedGary #B
    sepia[:,:,1] *= normalizedGary #G
    sepia[:,:,2] *= normalizedGary #R

    return np.array(sepia, np.uint8)


def saturation(imgSat):
    imgHSV = cv.cvtColor(imgSat, cv.COLOR_BGR2HSV)
    factor = 0.5
    imgHSV[:,:,0] = imgHSV[:,:,0] * factor #B
    imgHSV[:,:,1] = imgHSV[:,:,1] * factor #G
    #imgHSV[:,:,2] = imgHSV[:,:,2] * factor #R
    imgNew = cv.cvtColor(imgHSV, cv.COLOR_HSV2BGR)
    return imgNew

def vigniette(img):
    
    return 0


imgSrc = cv.imread('Utils/SetGame.png')
imgGray = cv.cvtColor(imgSrc, cv.COLOR_RGB2GRAY)
imgSepia = sepia(imgSrc)
imgSat = saturation(imgSepia)


cv.imshow("imgSrc",imgSrc)
cv.imshow("imgGray",imgGray)
cv.imshow("Sepia", imgSepia)
cv.imshow("Saturation", imgSat)
cv.waitKey(0)

cv.imwrite("C:/Users/annes/Documents/Studium/Aktuelle Module/BV2/Abgaben/AB01-Results/imgSrc.png", imgSrc)
cv.imwrite("C:/Users/annes/Documents/Studium/Aktuelle Module/BV2/Abgaben/AB01-Results/imgGray.png", imgGray)
cv.imwrite("C:/Users/annes/Documents/Studium/Aktuelle Module/BV2/Abgaben/AB01-Results/imgSepia.png", imgSepia)
cv.imwrite("C:/Users/annes/Documents/Studium/Aktuelle Module/BV2/Abgaben/AB01-Results/imgSat.png", imgSat)

#C:\Users\annes\Documents\Studium\Aktuelle Module\BV1\Abgaben\AB01 - Bilder
#Quellen:
#https://gist.github.com/FilipeChagasDev/bb63f46278ecb4ffe5429a84926ff812
#ChatGTP
#https://www.pinterest.de/pin/760897299516483533/