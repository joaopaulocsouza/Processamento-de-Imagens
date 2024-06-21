import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('image.png', cv.IMREAD_GRAYSCALE)

img = np.array(img,)
cv.imshow('image',img)
cv.waitKey(0)
cv.destroyAllWindows()