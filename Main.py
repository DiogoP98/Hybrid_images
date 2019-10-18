import sys
import cv2
import numpy as np
import os
from skimage import io, color
import MyConvolution

def main():
    image = cv2.imread('data/cat.bmp')
    kernel = np.array([[1/81]*9,[1/81]*9,[1/81]*9,[1/81]*9,[1/81]*9,[1/81]*9,[1/81]*9,[1/81]*9,[1/81]*9])
    convolved = MyConvolution.convolve(image, kernel)
    print(convolved.shape)
    #cv2.imwrite('data/cat_colour_1_81.bmp', convolved) 

if __name__ == '__main__':
	main()
