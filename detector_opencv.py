import matplotlib.pylab as plt
import cv2 as cv
import numpy as np
image=cv.imread('lane.jpeg')
image=cv.cvtColor(image,cv.COLOR_BGR2RGB)

#print(image.shape)
height=image.shape[0]
width=image.shape[1]

def draw_the_line(img,lines):
    img=np.copy(img)
    blank_image=np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv.line(blank_image,(x1,y1),(x2,y2),(255,255,0),thickness=2)
    img=cv.addWeighted(img,0.8,blank_image,1,0.0)
    return img


def region_of_interst(img,vertices):
    mask=np.zeros_like(img)
    #channel_count=img.shape[2]
    match_mask_color=255
    cv.fillPoly(mask,vertices,match_mask_color)
    mask_image=cv.bitwise_and(img,mask)
    return mask_image


region_of_interst_vertices=[(0,height),(width/2,height/2),(width,height)]
gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
canny=cv.Canny(gray,100,200)

cropped=region_of_interst(canny,np.array([region_of_interst_vertices],np.int32))
line1=cv.HoughLinesP(cropped,rho=6,theta=np.pi/60,threshold=160,lines=np.array([]),minLineLength=40,maxLineGap=20)


image_width_line=draw_the_line(image,line1)
plt.imshow(image_width_line)
plt.show()

