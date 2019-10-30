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
kernel_odd = np.array([[1/25]*5, [1/25]*5, [1/25]*5, [1/25]*5, [1/25]*5, [1/25]*5, [1/25]*5])

def main():
    image = cv2.imread('data/dog_gray.bmp')
    image2 = cv2.imread('data/cat_gray.bmp')
    hybrid1 = MyHybridImages.myHybridImages(image, 8, image2, 8)
    cv2.imwrite('results/hybrid_dog_cat_gray.bmp', hybrid1)

if __name__ == '__main__':
	main()
