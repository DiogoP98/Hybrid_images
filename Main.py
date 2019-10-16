import sys
import cv2
import numpy as np
from skimage import io, color
import MyConvolution

def main():
    image = cv2.imread('data/cat.bmp')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    convolved = MyConvolution.convolve(gray, kernel)
    cv2.imshow('image',convolved)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    

if __name__ == '__main__':
	main()
