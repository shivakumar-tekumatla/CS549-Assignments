# import cv2
# import matplotlib.pyplot as plt
# img = cv2.imread("Phase1/BSDS500/Images/1.jpg", cv2.IMREAD_GRAYSCALE)

# (thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# plt.imshow(img,cmap='gray')
# plt.show()

# cv2.imshow("Image",img)
# cv2.waitKey(0) & 0xFF 
# cv2.destroyAllwindows()
import math
import numpy as np
import cv2
def gaussian_kernel(scale,size):
    """Generating a gaussian filter of a given size and standard deviation
    Implemented as given in the below paper :
    https://pages.stat.wisc.edu/~mchung/teaching/MIA/reading/diffusion.gaussian.kernel.pdf.pdf"""
    pi = math.pi
    arr_x = [(i-size//2) for i in range(size)]
    arr_y = [(i-size//2) for i in range(size)]
    G2D = lambda x,y,std: (1/(2*pi*(std**2))) * math.exp(-((x**2)+(y**2))/(2*(std**2)))

    gaussian_arr = np.asarray([G2D(x,y,scale) for x in arr_x for y in arr_y])
    return gaussian_arr.reshape(size,size)

print(gaussian_kernel(1,5))

cv2.GaussianBlur()