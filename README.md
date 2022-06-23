# Image-Processing
SAMPLE-Progs
http://localhost:8889/tree/ImgProcessing_Sample%20-%20AllenPhilips

# 1.Program to display grayscale image using read and write operation.

import cv2<br>
img=cv2.imread('dog.jpg',0)<br>
cv2.imshow('dog',img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

OUTPUT:

![Grayscale](https://user-images.githubusercontent.com/98145104/173812852-eb93e44b-8173-49e1-9d47-96ee02db6739.png)

# 2.Program to display the image using matplotlib.

import matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>
img = mping.imread('Car.jpg')<br>
plt.imshow(img)<br>

OUTPUT:

![MatPlotLib](https://user-images.githubusercontent.com/98145104/173813054-896cb84f-29ab-492c-830d-4a1eb3a86c3d.png)

# 3.Program to perform linear transformation rotation.

import cv2<br>
from PIL import Image<br>
img=Image.open("vase.jpg")<br>
img=img.rotate(180)<br>
img.show()<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

OUTPUT:

![ImgRotate](https://user-images.githubusercontent.com/98145104/173812749-03040197-bc0e-44b9-a3ef-95ae5b41c649.png)

# 4.Program to convert color string to RGB color value.

from PIL import ImageColor<br>
#using getrgb for yellow<br>
img1=ImageColor.getrgb("yellow")<br>
print(img1)<br>
#using getrgb for red<br>
img2=ImageColor.getrgb("red")<br>
print(img2)<br>

OUTPUT:

![ColorToRGBvalues](https://user-images.githubusercontent.com/98145104/173813395-f006c078-0353-4d09-a93e-7ce48aeb2854.png)

# 5.Program to create image using colors.

from PIL import Image <br>
img=Image.new('RGB',(200,400),(255,255,0))<br>
img.show()<br>

OUTPUT:

![ImgUsingColors](https://user-images.githubusercontent.com/98145104/173813593-57ca8122-da6f-4a77-814d-73e936e30bde.png)

# 6.Program to visualize the image using various color spaces.

import cv2<br>
import matplotlib.pyplot as plt<br>
import numpy as np<br>
img=cv2.imread('ball.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)<br>
plt.imshow(img)<br>
plt.show()<br>

OUTPUT:

![ImgVisualization](https://user-images.githubusercontent.com/98145104/173814250-7c026e2f-3113-4914-9870-c3dd01c01d18.png)


# 7.Program to display the image attributes.

from PIL import Image<br>
image=Image.open('baby.jpg')<br>
print("Filename:", image.filename)<br>
print("Format:", image.format)<br>
print("Mode:", image.mode)<br>
print("Size:", image.size)<br>
print("Width:", image.width)<br>
print("Height:", image.height)<br>
image.close()<br>

OUTPUT:

![ImgAttributes](https://user-images.githubusercontent.com/98145104/173813911-28d7b1d8-074b-486b-b144-f4636cc36b50.png)

# 8.Program to convert original image to grayscale and binary.

import cv2<br>
#read the image file<br>
img=cv2.imread('sunflower.jpg')<br>
cv2.imshow('RGB',img)<br>
cv2.waitKey(0)<br>

#Gray scale

img=cv2.imread('sunflower.jpg',0)<br>
cv2.imshow('gray',img)<br>
cv2.waitKey(0)<br>

#Binary image

ret, bw_img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)<br>
cv2.imshow('binary', bw_img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

OUTPUT:

![rgb](https://user-images.githubusercontent.com/98145104/174041532-7a56dbcf-2208-4a93-bec1-8dc51c373f62.png)
![Grayscale](https://user-images.githubusercontent.com/98145104/174041573-f094cc0c-f80c-4004-8525-5744f6f0a146.png)
![binary](https://user-images.githubusercontent.com/98145104/174041610-aead2ff5-c608-44be-b43f-acb1b8ab7253.png)

# 9.Program to Resize the original image.

import cv2<br>
img=cv2.imread('pineapple.jpg')<br>
print('Length and Width of Original image', img.shape)<br>
cv2.imshow('original image',img)<br>
cv2.waitKey(0)<br>

#to show the resized image

imgresize=cv2.resize(img,(150,160))<br>
cv2.imshow('Resized image',imgresize)<br>
print('Length and Width of Resized image',imgresize.shape)<br>
cv2.waitKey(0)

OUTPUT:

![OrgIMG](https://user-images.githubusercontent.com/98145104/174042102-a5afbe04-6ad3-45bd-8b1e-5fd298da7b5e.png)

Length and Width of Original image (340, 453, 3)

![ResizedIMG](https://user-images.githubusercontent.com/98145104/174042157-26d460c4-d6b9-4242-83b0-cce5127f2f1e.png)

Length and Width of Resized image (160, 150, 3)

# 10.Develop a program to readimage using URL.
pip install Scikit-Learn<br>
import skimage<br>
print(skimage.__version__)<br>
start:
<br>
from skimage import io<br>
import matplotlib.pyplot as plt<br>
url='https://th.bing.com/th/id/OIP.qVuoApbCGfaPbNRSX8SmIwHaGA?w=213&h=180&c=7&r=0&o=5&pid=1.7.jpg'
image=io.imread(url)<br>
plt.imshow(image)<br>
plt.show()<br>
OUTPUT:
![URL](https://user-images.githubusercontent.com/98145104/175019845-d3b5671a-dc63-4e8c-9c50-fd807167b944.png)

# 11.Program to mask and blur the image.

import cv2<br>
import matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>
img=mping.imread('Orangef.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>

OUTPUT:
![Msk1](https://user-images.githubusercontent.com/98145104/175020316-23497261-1437-438c-8133-0e01451d77bc.png)

hsv_img=cv2.cvtColor(img, cv2.COLOR_RGB2HSV)<br>
light_orange=(1,190,200)<br>
dark_orange=(18,255,255)<br>
mask=cv2.inRange(hsv_img, light_green, dark_green)<br>
result=cv2.bitwise_and(img ,img, mask=mask)<br>
plt.subplot(1,2,1)<br>
plt.imshow(mask,cmap='gray')<br>
plt.subplot(1,2,2)<br>
plt.imshow(result)<br>
plt.show()<br>

OUTPUT:
![Msk2](https://user-images.githubusercontent.com/98145104/175020519-9b50e9e4-ee9c-4fdc-b3af-4cb76fc6be78.png)

light_white=(0,0,200)<br>
dark_white=(145,60,255)<br>
mask_white=cv2.inRange(hsv_img, light_white, dark_white)<br>
result_white=cv2.bitwise_and(img, img, mask=mask_white)<br>
plt.subplot(1,2,1)<br>
plt.imshow(mask_white, cmap='gray')<br>
plt.subplot(1,2,2)<br>
plt.imshow(result_white)<br>
plt.show()<br>

OUTPUT:
![Msk3](https://user-images.githubusercontent.com/98145104/175020627-d9aca11e-3bc7-4d0a-b90f-74e01fc71741.png)

final_mask = mask + mask_white<br>
final_result = cv2.bitwise_and(img, img, mask=final_mask)<br>
plt.subplot(1,2,1)<br>
plt.imshow(final_mask, cmap='gray')<br>
plt.subplot(1,2,2)<br>
plt.imshow(final_result)<br>
plt.show()<br>

OUTPUT:
![Msk4](https://user-images.githubusercontent.com/98145104/175020691-e09ca9d8-4a52-44f1-a20c-801ad12b5446.png)

blur = cv2.GaussianBlur(final_result, (7,7),0)<br>
plt.imshow(blur)<br>
plt.show()<br>

OUTPUT:
![Msk5](https://user-images.githubusercontent.com/98145104/175020753-75219b01-62ab-49bf-8210-c4f11e2520d3.png)

# 12.program to perform arithmatic operations on images.
import cv2<br>
import matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>

#Reading image files<br>
img1 = cv2.imread('Car.jpg')<br>
img2 = cv2.imread('Car1.jpg')<br>

#Applying Numpy addition on images<br>
fimg1 = img1 + img2<br>
plt.imshow(fimg1)<br>
plt.show()<br>

OUTPUT:
![Arith1](https://user-images.githubusercontent.com/98145104/175261389-c0b0a56f-70d2-46b7-81a1-33554053f777.png)

#Saving the output image<br>
cv2.imwrite('output.jpg', fimg1)<br>
fimg2 = img1 - img2<br>
plt.imshow(fimg2)<br>
plt.show()<br>

OUTPUT:
![Arith2](https://user-images.githubusercontent.com/98145104/175261566-dde86874-29f8-46ff-ba1f-01c61e7370ae.png)

#Saving the output image<br>
cv2.imwrite('output.jpg', fimg2)<br>
fimg3 = img1 * img2<br>
plt.imshow(fimg3)<br>
plt.show()<br>

OUTPUT:
![Arith3](https://user-images.githubusercontent.com/98145104/175262138-4241d2f9-b285-41fb-89dd-ccbe1f48ff92.png)

#Saving the output image<br>
cv2.imwrite('output.jpg', fimg3)<br>
fimg4 = img1 / img2<br>
plt.imshow(fimg4)<br>
plt.show()<br>

#Saving the output image<br>
cv2.imwrite('output.jpg', fimg4)<br>

OUTPUT:
![Arith4](https://user-images.githubusercontent.com/98145104/175262202-5cf609ad-c8dd-4ff9-bbbd-cd433bd06e3e.png)

# 13.Program to change the image to different color spaces.

import cv2<br>
img = cv2.imread('D:\\img.jpg')<br>
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)<br>
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)<br>
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)<br>
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)<br>
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)<br>
cv2.imshow("GRAY image", gray)<br>
cv2.imshow("HSV image",hsv)<br>
cv2.imshow("LAB image",lab)<br>
cv2.imshow("HLS image", hls)<br>
cv2.imshow("YUV image",yuv)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

OUTPUT:
![Gray](https://user-images.githubusercontent.com/98145104/175279385-b0124626-1674-4676-b879-ff4d371d072c.png)
![HSV](https://user-images.githubusercontent.com/98145104/175279426-29f54544-831e-4121-a85e-ff444cc3986e.png)
![LAB](https://user-images.githubusercontent.com/98145104/175279456-9f01e5ff-ccfa-4d16-88f2-03ff73c6b7d4.png)
![HLS](https://user-images.githubusercontent.com/98145104/175279520-ad740d11-c093-4a24-a60d-2d88811068b1.png)
![YUV](https://user-images.githubusercontent.com/98145104/175279538-0de4b362-03d6-4fcb-acd4-c503d6d36c61.png)

# 14.Program to create an image using 2D array.

import cv2 as c<br>
import numpy as np<br>
from PIL import Image<br>
array = np.zeros([100,200,3], dtype = np.uint8)<br>
array[:,:100] = [255,130,0]<br>
array[:,100:] = [0,0,255]<br>
img = Image.fromarray(array)<br>
img.save('image1.png')<br>
img.show()<br>
c.waitKey(0)<br>

OUTPUT:![2D](https://user-images.githubusercontent.com/98145104/175268327-7a2e9d59-60a9-4fdb-8039-1bd29e06b282.png)
