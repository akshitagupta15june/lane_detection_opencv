import matplotlib.pylab as plt
import cv2 as cv
import numpy as np
image=cv.imread('lane.jpeg')
image=cv.cvtColor(image,cv.COLOR_BGR2RGB)

print(image.shape)
height=image.shape[0]
width=image.shape[1]
region_of_interst_vertices=[(0,height),(width/2,height/2),(width,height)]
def region_of_interst(img,vertices):
    mask=np.zeros_like(img)
    channel_count=img.shape[2]
    match_mask_color=(255,)*channel_count
    cv.fillPoly(mask,vertices,match_mask_color)
    mask_image=cv.bitwise_and(img,mask)
    return mask_image
cropped=region_of_interst(image,np.array([region_of_interst_vertices],np.int32))
plt.imshow(cropped)
plt.show()

