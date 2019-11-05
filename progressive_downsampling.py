import cv2
 
img = cv2.imread('/Users/Diogo/Desktop/MSc_AI/Computer Vision/Hybrid_images/results/hybrid_boris_trump_5.bmp', cv2.IMREAD_UNCHANGED)
 
print('Original Dimensions : ',img.shape)
 
scale_percent = 65 # percent of original size

for i in range(4):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite("results/downsample" + str(i) + ".bmp",  img)
