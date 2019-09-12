'''

Phase-correlation with log-polar transform for calculating
rotational shift between two images.

'''

#dependencies - can be installed using pip or conda
import cv2
import numpy as np

#read image from path - edit this parameter to point to image
base_img = cv2.imread('184-C-3201905251028114.jpg')
#convert loaded image to necessary format
base_img = np.float32(cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)) / 255.0

#store image shape, and image centre
(h, w) = base_img.shape
(cX, cY) = (w // 2, h // 2)

#determine rotation parameters for new rotated image
scale_percent = 40 # percent of original size
width = int(base_img.shape[1] * scale_percent / 100) 
height = int(base_img.shape[0] * scale_percent / 100) 
dim = (width, height) 

#define degree of rotation for rotated image
angle = 38

#generate rotated image
M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
curr_img = cv2.warpAffine(base_img, M, (w, h))

# resize output window
resized = cv2.resize(base_img, dim, interpolation = cv2.INTER_AREA) 
resized2 = cv2.resize(curr_img, dim, interpolation = cv2.INTER_AREA) 

#show original image and resized image
cv2.imshow("base_img", resized)
cv2.imshow("curr_img", resized2)

#generate log-polar versions of original and rotated image
base_polar = cv2.linearPolar(base_img,(cX, cY), min(cX, cY), 0)
curr_polar = cv2.linearPolar(curr_img,(cX, cY), min(cX, cY), 0) 

#resize output window
resizedp = cv2.resize(base_polar, dim, interpolation = cv2.INTER_AREA) 
resizedp2 = cv2.resize(curr_polar, dim, interpolation = cv2.INTER_AREA) 

#show log-polar images
cv2.imshow("base_polar", resizedp)
cv2.imshow("curr_polar", resizedp2)

#get shift between polar images using phase correlation
(sx, sy), sf = cv2.phaseCorrelate(base_polar, curr_polar)

#convert shift to rotation in degrees
rotation = -sy / h * 360;
#print rotation
print(rotation) 

#clean up
cv2.waitKey(0)
cv2.destroyAllWindows()