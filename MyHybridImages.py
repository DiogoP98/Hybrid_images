import math
import numpy as np
import cv2

from MyConvolution import convolve


def myHybridImages(lowImage: np.ndarray, lowSigma: float, highImage: np.ndarray, highSigma: float) -> np.ndarray:
    """
    Create hybrid images by combining a low-pass and high-pass filtered pair.
    
    :param lowImage: the image to low-pass filter (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
    :type numpy.ndarray
    
    :param lowSigma: the standard deviation of the Gaussian used for low-pass filtering lowImage
    :type float
    
    :param highImage: the image to high-pass filter (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
    :type numpy.ndarray
    
    :param highSigma: the standard deviation of the Gaussian used for low-pass filtering highImage before subtraction to create the high-pass filtered image
    :type float 
    
    :returns returns the hybrid image created
           by low-pass filtering lowImage with a Gaussian of s.d. lowSigma and combining it with 
           a high-pass image created by subtracting highImage from highImage convolved with
           a Gaussian of s.d. highSigma. The resultant image has the same size as the input images.
    :rtype numpy.ndarray
    """

    lowImage = lowImage.astype(float)
    highImage = highImage.astype(float)
    lowImage /= 255
    highImage /= 255

    low_freq_kernel = makeGaussianKernel(lowSigma)
    low_pass = convolve(lowImage,low_freq_kernel)

    high_freq_kernel = makeGaussianKernel(highSigma)
    high_pass = highImage - convolve(highImage,high_freq_kernel)

    cv2.imwrite('results/dog_lowpass_gray.bmp',  low_pass * 255)
    cv2.imwrite('results/cat_highpass_gray.bmp', (high_pass + 0.5) * 255)

    return (low_pass + high_pass) * 255
    

def makeGaussianKernel(sigma: float) -> np.ndarray:
    """
    Use this function to create a 2D gaussian kernel with standard deviation sigma.
    The kernel values should sum to 1.0, and the size should be floor(8*sigma+1) or 
    floor(8*sigma+1)+1 (whichever is odd) as per the assignment specification.
    """

    size = int(np.floor(8.0 * sigma + 1.0))
    if size % 2 == 0:
        size+= 1; # size must be odd
    
    kernel = np.empty([size,size], dtype = float)

    centre = int(np.floor(size / 2))
    sigma_squared = sigma**2
    sum = 0

    for x in range(size):
        for y in range(size):
            numerator = -((y - centre) ** 2 + (x - centre) ** 2)
            kernel[x,y] = np.exp(numerator / sigma_squared)
            sum += kernel[x,y]
    
    #normalize kernel
    kernel /= sum

    return kernel
