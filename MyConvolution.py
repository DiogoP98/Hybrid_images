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
	
	if len(image.shape) == 3: #colour image
		colour_image = True
		imagerows, imagecols, channels = image.shape
		convolution = np.zeros((imagerows, imagecols, channels), dtype = float)
	else: #gray image
		imagerows, imagecols = image.shape
		convolution = np.zeros((imagerows, imagecols), dtype = float)
	
	kernelrows, kernelcols = kernel.shape

	#get size of each border for image padding
	xborder = int(np.floor(kernelrows/2))
	yborder = int(np.floor(kernelcols/2))

	#Create image with borders
	if colour_image:
		image_padding = np.zeros((imagerows+xborder*2, imagecols+yborder*2, channels))
	else:
		image_padding = np.zeros((imagerows+xborder*2, imagecols+yborder*2))
	
	image_padding[xborder:-xborder,yborder:-yborder] = image

	#flip the kernel
	kernel = np.flip(kernel, axis = 0)
	kernel = np.flip(kernel, axis = 1)

	for x in range(xborder, imagerows + xborder):
		for y in range(yborder, imagecols + yborder):
			if colour_image:
				for colour in range(3): #go through each colour channel
					convolution[x - xborder, y - yborder, colour] = np.multiply(kernel,image_padding[x - xborder: x + xborder + 1, y - yborder: y + yborder + 1, colour]).sum()
			else:
	 			convolution[x- xborder, y - yborder] = np.multiply(kernel,image_padding[x - xborder: x + xborder + 1, y - yborder: y + yborder + 1]).sum()
	
	return convolution
