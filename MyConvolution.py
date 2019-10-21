import numpy as np

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
	
	colour_image = False
	
	if len(image.shape) == 3: #colour
		colour_image = True
		imagerows, imagecols, channels = image.shape
		convolution = np.zeros((imagerows, imagecols, channels), dtype = float)
	else: 
		imagerows, imagecols = image.shape
		convolution = np.zeros((imagerows, imagecols), dtype = float)
	
	kernelrows, kernelcols = kernel.shape

	xborder = int(np.floor(kernelrows/2))
	yborder = int(np.floor(kernelcols/2))

	#Create image with borders
	if colour_image:
		image_padding = np.zeros((imagerows+xborder*2, imagecols+yborder*2, channels))
	else:
		image_padding = np.zeros((imagerows+xborder*2, imagecols+yborder*2))
	
	image_padding[xborder:-xborder,yborder:-yborder] = image

	kernel = np.flip(kernel, axis = 0)
	kernel = np.flip(kernel, axis = 1)

	for x in range(yborder + 1, imagecols - yborder + 1):
		for y in range(xborder + 1, imagerows - xborder + 1):
			if colour_image:
				for colour in range(3):
					convolution[y, x, colour] = (kernel * image_padding[y - xborder - 1: y + kernelrows - xborder - 1, x - yborder - 1: x + kernelcols - yborder - 1, colour]).sum()
			else:
	 			convolution[y, x] = (kernel * image_padding[y - xborder: y + kernelrows - xborder, x - yborder: x + kernelcols - yborder]).sum()

	return convolution

def fourier_convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
	"""
	Convolve an image with a kernel assuming zero-padding of the image to handle the borders
	
	:param image: the image (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
	:type numpy.ndarray
	
	:param kernel: the kernel (shape=(kheight,kwidth); both dimensions odd)
	:type numpy.ndarray 
	
	:returns the convolved image (of the same shape as the input image)
	:rtype numpy.ndarray
	"""
	
	colour_image = False
	
	if len(image.shape) == 3: #colour
		colour_image = True
		imagerows, imagecols, channels = image.shape
		convolution = np.zeros((imagerows, imagecols, channels), dtype = float)
	else: 
		imagerows, imagecols = image.shape
		convolution = np.zeros((imagerows, imagecols), dtype = float)
	
	kernelrows, kernelcols = kernel.shape

	xborder = int(np.floor(kernelrows/2))
	yborder = int(np.floor(kernelcols/2))

	widthpad, heightPad = xborder * 2 + kernelrows, yborder * 2 + kernelcols

	#Create kernel with borders
	kernel_padding = np.zeros((widthpad, heightPad))

	kernel = np.flip(kernel, axis = 0)
	kernel = np.flip(kernel, axis = 1)

	kernel_padding[xborder: -xborder, yborder: -yborder] = kernel

	imageTransform = np.fft.fft2(image)
	kernelTransform = np.fft.fft2(kernel_padding)

	return convolution
	
