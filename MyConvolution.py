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

	xborder = int(np.floor((imagerows - kernelrows)/2))
	yborder = int(np.floor((imagecols - kernelcols)/2))

	#flip kernel
	kernel = np.flip(kernel, axis = 0)
	kernel = np.flip(kernel, axis = 1)

	#create kernel with same size as image
	kernel_padding = np.zeros([imagerows, imagecols])
	kernel_padding[xborder:xborder + kernelrows, yborder: yborder + kernelcols] = kernel

	convolution = np.empty_like(image)

	if not colour_image:
		imageTransform = np.fft.rfft2(image)
		kernelTransform = np.fft.rfft2(kernel_padding)
		convolution = np.fft.irfft2(imageTransform * kernelTransform)
	else:
		for colours in range(3):
			imageTransform = np.fft.rfft2(image[:,:,colours])
			kernelTransform = np.fft.rfft2(kernel_padding)
			convolution[:,:,colours] = np.fft.fftshift(np.fft.irfft2(imageTransform * kernelTransform))

	return convolution
	
