import numpy as np

def convolve_function(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
	convolution = np.zeros_like(image)

	imagedim = image.shape
	kerneldim = kernel.shape

	xborder = int(np.floor(kerneldim[0]/2))
	yborder = int(np.floor(kerneldim[1]/2))

	zeros_x_vector = np.zeros((image.shape[0]))
	zeros_y_vector = np.zeros((image.shape[1]))
	#print(image)


	image_padding = np.zeros((imagedim[0]+xborder, imagedim[1]+yborder))
	
	for x in range(imagedim[0]):
		for y in range(imagedim[1]):
			image_padding[x+int(xborder/2),y+int(yborder/2)] = image[x][y] 
	print(image_padding)

	for x in range(yborder + 1, imagedim[1] - yborder):
		for y in range(xborder + 1, imagedim[0] - xborder):
			convolution[y, x] = (kernel * image_padding[y:y + kerneldim[1], x:x + kerneldim[0]]).sum()
	
	#print(convolution)

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
		return convolve_function(image, kernel)
	elif len(image.shape) == 3: #colour
		print(image[:,:,0].shape)
		convolved_channel1 = convolve_function(image[:,:,0], kernel)
		convolved_channel2 = convolve_function(image[:,:,1], kernel)
		convolved_channel3 = convolve_function(image[:,:,2], kernel)

		#print(convolved_channel1)
		convolved_image = np.zeros((image.shape[0], image.shape[1], image.shape[2]))

		for x in range(image.shape[0]):
			for y in range(image.shape[1]):
				convolved_image[x,y] = [convolved_channel1[x][y],convolved_channel2[x][y],convolved_channel3[x][y]]
		
		return convolved_image
	
