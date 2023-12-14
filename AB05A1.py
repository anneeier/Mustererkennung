import cv2
import numpy as np

# Lade das große Bild und das kleine Teilbild
large_image = cv2.imread('Vildkatten/Vildkatten.jpg', cv2.IMREAD_GRAYSCALE)
small_image1 = cv2.imread('Vildkatten/VildkattenKarte01.png', cv2.IMREAD_GRAYSCALE)
small_image2 = cv2.imread('Vildkatten/VildkattenKarte02.png', cv2.IMREAD_GRAYSCALE)
small_image3 = cv2.imread('Vildkatten/VildkattenKarte03.png', cv2.IMREAD_GRAYSCALE)
# small_image4 = cv2.imread('Vildkatten/VildkattenKarte04.png', cv2.IMREAD_GRAYSCALE)
# small_image5 = cv2.imread('Vildkatten/VildkattenKarte05.png', cv2.IMREAD_GRAYSCALE)
# small_image6 = cv2.imread('Vildkatten/VildkattenKarte06.png', cv2.IMREAD_GRAYSCALE)
# small_image7 = cv2.imread('Vildkatten/VildkattenKarte07.png', cv2.IMREAD_GRAYSCALE)

# Initialisiere den SIFT-Detektor
sift = cv2.SIFT_create()

# Finde die Key-Points und Descriptoren mit SIFT für beide Bilder
keypoints1, descriptors1 = sift.detectAndCompute(small_image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(small_image2, None)
keypoints3, descriptors3 = sift.detectAndCompute(small_image3, None)
keypointsL, descriptorsL = sift.detectAndCompute(large_image, None)

# Initialisiere den Brute-Force-Matcher
bf = cv2.BFMatcher()

# Finde die besten Übereinstimmungen zwischen den Descriptoren der beiden Bilder
matches = bf.knnMatch( descriptors1, descriptorsL, k=2)

# Anwenden des Ratio-Tests, um die besten Übereinstimmungen zu filtern
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Konvertiere Key-Points in das Format, das RANSAC erwartet
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypointsL[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Berechne die affine Transformation mit RANSAC
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0, None,)

# Transformiere die Ecken des kleinen Bildes
h, w = small_image1.shape
corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
transformed_corners = cv2.perspectiveTransform(corners, M)

# Zeichne das Rechteck um den gefundenen Ausschnitt im großen Bild
large_image_with_rectangle = large_image.copy()
cv2.polylines(large_image_with_rectangle, [np.int32(transformed_corners)], True, (0, 0, 0), 15)

# Zeige die Ergebnisse an
cv2.imshow('Large Image with Rectangle', large_image_with_rectangle)
cv2.imwrite('output_image.png', large_image_with_rectangle)
cv2.waitKey(0)
cv2.destroyAllWindows()

