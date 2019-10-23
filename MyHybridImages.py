import math
import numpy as np
import cv2

from MyConvolution import convolve
from MyConvolution import fourier_convolve


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

    fourier = True

    low_freq_kernel = makeGaussianKernel(lowSigma)
    if fourier:
        low_pass = fourier_convolve(lowImage,low_freq_kernel)
    else:
        low_pass = convolve(lowImage,low_freq_kernel)

    high_freq_kernel = makeGaussianKernel(highSigma)

    if fourier:
        high_pass = highImage - fourier_convolve(highImage,high_freq_kernel)
    else:
        xborder_high = int(np.floor(high_freq_kernel.shape[0]/2))
        yborder_high = int(np.floor(high_freq_kernel.shape[1]/2))
        
        #put black border
        high_pass = np.zeros(highImage.shape)
        diff = highImage[xborder_high + 1:-xborder_high - 1,yborder_high + 1:-yborder_high - 1] - convolve(highImage,high_freq_kernel)[xborder_high + 1:-xborder_high - 1,yborder_high + 1:-yborder_high - 1]
        high_pass[xborder_high + 1:-xborder_high - 1,yborder_high + 1:-yborder_high - 1] = diff
    
    cv2.imwrite('results/lowpass_dog_fourier.bmp', low_pass)
    cv2.imwrite('results/highpass_cat_fourier.bmp', high_pass + 0.5) 

    return low_pass + high_pass
    

def makeGaussianKernel(sigma: float) -> np.ndarray:
    """
    Use this function to create a 2D gaussian kernel with standard deviation sigma.
    The kernel values should sum to 1.0, and the size should be floor(8*sigma+1) or 
    floor(8*sigma+1)+1 (whichever is odd) as per the assignment specification.
    """

    size = int(np.floor(8.0 * sigma + 1.0)); # (this implies the window is +/- 4 sigmas from the centre of the Gaussian)
    if size % 2 == 0:
        size+= 1; # size must be odd
    
    kernel = np.empty([size,size], dtype = float)

    sigma_squared = float(sigma**2)
    multiplier = float(1/(2*math.pi*sigma_squared))
    divider = float(2*sigma_squared)

    sum = 0.0
    for x in range(size):
         for y in range(size):
             kernel[x,y] = float(multiplier * math.e**(-(x**2+y**2)/divider))
             sum += kernel[x,y]
    
    #normalize kernel
    kernel /= sum

    return kernel
