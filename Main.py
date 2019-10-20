import sys
import cv2
import numpy as np
import os
from skimage import io, color
import MyConvolution
import MyHybridImages

kernel_3 = np.array([[1/9]*3, [1/9]*3, [1/9]*3])
kernel_5 = np.array([[1/25]*5, [1/25]*5, [1/25]*5, [1/25]*5, [1/25]*5])
kernel_9 = np.array([[1/81]*9,[1/81]*9,[1/81]*9,[1/81]*9,[1/81]*9,[1/81]*9,[1/81]*9,[1/81]*9,[1/81]*9])

def main():
    image = cv2.imread('data/einstein.bmp')
    image2 = cv2.imread('data/marilyn.bmp')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    #convolved = MyConvolution.convolve(gray, kernel_9)
    hybrid1 = MyHybridImages.myHybridImages(gray, 1.4, gray2, 1.2)
    #print(kernel)
    #print(convolved.shape)
    #print(gray.shape)
    cv2.imwrite('data/hybrid_2.bmp', hybrid1) 

if __name__ == '__main__':
	main()
