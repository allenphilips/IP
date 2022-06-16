# Image-Processing
SAMPLE-Progs
http://localhost:8889/tree/ImgProcessing_Sample%20-%20AllenPhilips

1.Program to display grayscale image using read and write operation.

import cv2 
img=cv2.imread('dog.jpg',0)
cv2.imshow('dog',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

OUTPUT:
![Grayscale](https://user-images.githubusercontent.com/98145104/173812852-eb93e44b-8173-49e1-9d47-96ee02db6739.png)

2.Program to display the image using matplotlib.

import matplotlib.image as mping
import matplotlib.pyplot as plt
img = mping.imread('Car.jpg')
plt.imshow(img)

OUTPUT:
![MatPlotLib](https://user-images.githubusercontent.com/98145104/173813054-896cb84f-29ab-492c-830d-4a1eb3a86c3d.png)

3.Program to perform linear transformation rotation.

import cv2
from PIL import Image
img=Image.open("vase.jpg")
img=img.rotate(180)
img.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

OUTPUT:
![ImgRotate](https://user-images.githubusercontent.com/98145104/173812749-03040197-bc0e-44b9-a3ef-95ae5b41c649.png)

4.Program to convert color string to RGB color value.

from PIL import ImageColor
#using getrgb for yellow
img1=ImageColor.getrgb("yellow")
print(img1)
#using getrgb for red
img2=ImageColor.getrgb("red")
print(img2)

OUTPUT:
![ColorToRGBvalues](https://user-images.githubusercontent.com/98145104/173813395-f006c078-0353-4d09-a93e-7ce48aeb2854.png)

5.Program to create image using colors.

from PIL import Image 
img=Image.new('RGB',(200,400),(255,255,0))
img.show()

OUTPUT:
![ImgUsingColors](https://user-images.githubusercontent.com/98145104/173813593-57ca8122-da6f-4a77-814d-73e936e30bde.png)

6.Program to visualize the image using various color spaces.

import cv2
import matplotlib.pyplot as plt
import numpy as np
img=cv2.imread('ball.jpg')
plt.imshow(img)
plt.show()
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
plt.imshow(img)
plt.show()

OUTPUT:
![ImgVisualization](https://user-images.githubusercontent.com/98145104/173814250-7c026e2f-3113-4914-9870-c3dd01c01d18.png)


7.Program to display the image attributes.

from PIL import Image
image=Image.open('baby.jpg')
print("Filename:", image.filename)
print("Format:", image.format)
print("Mode:", image.mode)
print("Size:", image.size)
print("Width:", image.width)
print("Height:", image.height)
image.close()

OUTPUT:
![ImgAttributes](https://user-images.githubusercontent.com/98145104/173813911-28d7b1d8-074b-486b-b144-f4636cc36b50.png)

8.Program to convert original image to grayscale and binary.

import cv2
#read the image file
img=cv2.imread('sunflower.jpg')
cv2.imshow('RGB',img)
cv2.waitKey(0)

#Gray scale
img=cv2.imread('sunflower.jpg',0)
cv2.imshow('gray',img)
cv2.waitKey(0)

#Binary image
ret, bw_img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
cv2.imshow('binary', bw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

OUTPUT:
![rgb](https://user-images.githubusercontent.com/98145104/174041532-7a56dbcf-2208-4a93-bec1-8dc51c373f62.png)
![Grayscale](https://user-images.githubusercontent.com/98145104/174041573-f094cc0c-f80c-4004-8525-5744f6f0a146.png)
![binary](https://user-images.githubusercontent.com/98145104/174041610-aead2ff5-c608-44be-b43f-acb1b8ab7253.png)

9.Program to Resize the original image.

import cv2

#read the image file
img=cv2.imread('sunflower.jpg')
cv2.imshow('RGB',img)
cv2.waitKey(0)

#Gray scale
img=cv2.imread('sunflower.jpg',0)
cv2.imshow('gray',img)
cv2.waitKey(0)

#Binary image
ret, bw_img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
cv2.imshow('binary', bw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

OUTPUT:

![OrgIMG](https://user-images.githubusercontent.com/98145104/174042102-a5afbe04-6ad3-45bd-8b1e-5fd298da7b5e.png)

Length and Width of Original image (340, 453, 3)

![ResizedIMG](https://user-images.githubusercontent.com/98145104/174042157-26d460c4-d6b9-4242-83b0-cce5127f2f1e.png)

Length and Width of Resized image (160, 150, 3)
