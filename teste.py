from PIL import Image, ImageChops 
  
# Opening Images 
im = Image.open(r"/Users/Diogo/Desktop/MSc_AI/Computer Vision/Hybrid_images/data/boris_22.bmp")
  
# Here, xoffset is given 100 
# yoffset wil automaticallly set to 100 
#
im3 = ImageChops.offset(im, 10, 5) 
  
# showing resultant image 
im3.save("/Users/Diogo/Desktop/MSc_AI/Computer Vision/Hybrid_images/data/boris_23.bmp", "bmp") 
