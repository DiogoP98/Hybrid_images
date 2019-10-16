import numpy as np

def convolve_gray(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
	convolution = np.zeros_like(image)

	imagedim = image.shape
	kerneldim = kernel.shape

	xborder = int(np.floor(kerneldim[0]/2))
	yborder = int(np.floor(kerneldim[1]/2))

	#add borders
	image = np.insert(image, (0, imagedim[0]), 0, axis = 0) 
	image = np.insert(image, (0, imagedim[1]), 0, axis = 1)

	for x in range(yborder + 1, imagedim[1] - yborder):
		for y in range(xborder + 1, imagedim[0] - xborder):
			convolution[y, x] = (kernel * image[y:y + 3, x:x + 3]).sum()
 
	return convolution

def convolve_colour(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
	convolution = np.zeros_like(image)
	return np.array()

#width of the border equals half the size of the template
def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
	"""
	Convolve an image with a kernel assuming zero-padding of the image to handle the borders
	
	:param image: the image (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
	:type numpy.ndarray
	
	:param kernel: the kernel (shape=(kheight,kwidth); both dimensions odd)
	:type numpy.ndarray 
	
	:returns the convolved image (of the same shape as the input image)
	:rtype numpy.ndarray
	"""
	# Your code here. You'll need to vectorise your implementation to ensure it runs
	# at a reasonable speed.

	if len(image.shape) < 3: #gray
		return convolve_gray(image, kernel)
	elif len(image.shape) == 3: #colour
		return convolve_colour(image, kernel)
	
