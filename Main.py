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
    image = cv2.imread('data/dog.bmp')
    image2 = cv2.imread('data/cat.bmp')
    #x = MyConvolution.fourier_convolve(image, kernel_9)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite('data/einstein_gray.bmp', gray)
    #cv2.imwrite('data/marilyn_gray.bmp', gray2) 
    #convolved = MyConvolution.convolve(image, test_kernel)
    #kernel = MyHybridImages.makeGaussianKernel(1.2)
    #cv2.imwrite('results/einstein_gray_convolved.bmp', MyConvolution.convolve(gray, kernel))
    #cv2.imwrite('results/marilyn_gray_convolved.bmp', MyConvolution.convolve(gray2, kernel))
    hybrid1 = MyHybridImages.myHybridImages(image, 4, image2, 4)
    #print(kernel)
    #print(convolved.shape)
    #print(gray.shape)
    cv2.imwrite('results/hybrid.bmp', hybrid1)

if __name__ == '__main__':
	main()
