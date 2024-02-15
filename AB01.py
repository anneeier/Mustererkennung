import cv2 as cv
import numpy as np
import math 

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


def vigniette(imgSrc, radius):    
    width, height, color = imgSrc.shape
    imgSrc = imgSrc/255
    result = imgSrc
    max_len = math.sqrt(width//2 * width//2 + height//2 * height//2)
    for y in range(height):
        for x in range(width):
            tmp_x = width//2 - x
            tmp_y = height//2 - y
            tmp_len = math.sqrt(tmp_x*tmp_x + tmp_y*tmp_y) 
            if tmp_len > radius:
                reduceBrigt = (tmp_len - radius)/ max_len
                result[x,y] = imgSrc[x,y] - max(0, reduceBrigt) 
    return result

import cv2
import numpy as np

def create_vignette(image):
    height, width = image.shape[:2]
    
    # Erzeuge eine Maske für die Vignette
    mask = np.zeros((height, width), dtype=np.uint8)
    cv.circle(mask, (width//2, height//2), min(width, height)//3, 255, -1)

    cv2.imshow('Mask ohne Filter', mask)

    
    # Erzeuge einen radialen Weichzeichnungseffekt
    blurred_mask = cv2.GaussianBlur(mask,(0,0),sigmaX=min(width, height)//7.5)

    cv2.imshow('Mask mit Filter', blurred_mask)

    # Teile das Bild in die Farbkanäle auf
    b, g, r = cv.split(image)

    # Wende die Vignette auf jeden Farbkanal an, indem du sie mit dem entsprechenden Farbkanal multiplizierst
    b_vignette = cv.multiply(b.astype(float), blurred_mask.astype(float) / 255)
    g_vignette = cv.multiply(g.astype(float), blurred_mask.astype(float) / 255)
    r_vignette = cv.multiply(r.astype(float), blurred_mask.astype(float) / 255)

    # Kombiniere die Farbkanäle zu einem Bild mit der Vignette
    vignette_image = cv.merge((b_vignette.astype(np.uint8), g_vignette.astype(np.uint8), r_vignette.astype(np.uint8)))
    return vignette_image

# Lade das Bild
image = cv2.imread('Utils/LennaCol.png')

# Erzeuge die Vignette
vignette_image = create_vignette(image)

# Zeige das Ergebnis
cv2.imshow('Originalbild', image)
cv2.imshow('Vignettenbild', vignette_image)
cv2.waitKey(0)
cv2.destroyAllWindows()




# imgSrc = cv.imread('Utils/LennaCol.png')

#img_white = np.ones((500,500))
# imgGray = cv.cvtColor(imgSrc, cv.COLOR_RGB2GRAY)
# imgSepia = sepia(imgSrc)
# imgSat = saturation(imgSepia)
# imgVig = vigniette(imgSat, 150)

# cv.imshow("Src",imgSrc)
# cv.imshow("Gray",imgGray)
# cv.imshow("Sepia", imgSepia)
# cv.imshow("Saturation", imgSat)
# cv.imshow('Vignette', imgVig)
# cv.waitKey(0)


#cv.imwrite("C:/Users/annes/Documents/Studium/Aktuelle Module/BV2/Mustererkennung/AB01-Results/imgSrc.png", imgSrc)
#cv.imwrite("C:/Users/annes/Documents/Studium/Aktuelle Module/BV2/Mustererkennung/AB01-Results/imgGray.png", imgGray)
#cv.imwrite("C:/Users/annes/Documents/Studium/Aktuelle Module/BV2/Mustererkennung/AB01-Results/imgSepia.png", imgSepia)
#cv.imwrite("C:/Users/annes/Documents/Studium/Aktuelle Module/BV2/Mustererkennung/AB01-Results/imgSat.png", imgSat)
#cv.imwrite("C:/Users/annes/Documents/Studium/Aktuelle Module/BV2/Mustererkennung/AB01-Results/imgVig.png", imgVig)


#C:\Users\annes\Documents\Studium\Aktuelle Module\BV1\Abgaben\AB01 - Bilder
#Quellen:
#https://gist.github.com/FilipeChagasDev/bb63f46278ecb4ffe5429a84926ff812
#ChatGTP
#https://www.pinterest.de/pin/760897299516483533/
