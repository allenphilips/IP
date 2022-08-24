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
cv2.waitKey(0)<br>

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
url='https://th.bing.com/th/id/OIP.qVuoApbCGfaPbNRSX8SmIwHaGA?w=213&h=180&c=7&r=0&o=5&pid=1.7.jpg'<br>
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

# 15.Program to perform bitwise operations on an image.

import cv2<br>
import matplotlib.pyplot as plt<br>
image1=cv2.imread('bird1.jpg')<br>
image2=cv2.imread('bird1.jpg')<br>
ax=plt.subplots(figsize=(15,10))<br>
bitwiseAnd  = cv2.bitwise_and(image1,image2)<br>
bitwiseOr   = cv2.bitwise_or(image1,image2)<br>
bitwiseXor  = cv2.bitwise_xor(image1,image2)<br>
bitwiseNot_img1 = cv2.bitwise_not(image1)<br>
bitwiseNot_img2 = cv2.bitwise_not(image2)<br>
plt.subplot(151)<br>
plt.imshow(bitwiseAnd)<br>
plt.subplot(152)<br>
plt.imshow(bitwiseOr)<br>
plt.subplot(153)<br>
plt.imshow(bitwiseXor)<br>
plt.subplot(154)<br>
plt.imshow(bitwiseNot_img1)<br>
plt.subplot(155)<br>
plt.imshow(bitwiseNot_img2)<br>
cv2.waitKey(0)<br>

OUTPUT:![Bitwise](https://user-images.githubusercontent.com/98145104/176425447-f6bff98a-4992-4abc-b7cd-2a21a6d8987e.png)

# 16.Program to perform blur operations.
import cv2<br>
import numpy as np<br>
image = cv2.imread('glass1.jpg')<br>
cv2.imshow('Original Image', image)<br>
cv2.waitKey(0)<br>

Gaussian = cv2.GaussianBlur(image, (7, 7),0)<br>
cv2.imshow('Gaussian Blurring', Gaussian)<br>
cv2.waitKey(0)<br>

median = cv2.medianBlur(image, 5)<br>
cv2.imshow('Median Blurring', median)<br>
cv2.waitKey(0)<br>

bilateral = cv2.bilateralFilter(image, 9, 75, 75)<br>
cv2.imshow('Bilateral Blurring', bilateral)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

OUTPUT:
![Blur1](https://user-images.githubusercontent.com/98145104/176425864-30afa493-aafa-4aad-877d-adc4aedf783b.png)
![Blur2](https://user-images.githubusercontent.com/98145104/176425891-e960c7c0-ade4-447e-94a6-f7149ca0c305.png)
![Blur3](https://user-images.githubusercontent.com/98145104/176425906-a2a3ee45-1a2b-4695-96c1-0251f46752d0.png)
![Blur4](https://user-images.githubusercontent.com/98145104/176425922-86ef55d4-c240-431e-acd6-81d5b1d7ab16.png)

# 17.Program to enhance an image.
from PIL import Image<br>
from PIL import ImageEnhance<br>
image = Image.open('bird2.jpg')<br>
image.show()<br>

#Brightness<br>
enh_bri = ImageEnhance.Brightness(image)<br>
brightness = 1.5<br>
image_brightened = enh_bri.enhance(brightness)<br>
image_brightened.show()<br>

#Color<br>
enh_col = ImageEnhance.Color(image)<br>
color = 1.5<br>
image_colored = enh_col.enhance(color)<br>
image_colored.show()<br>

#Contrast<br>
enh_con = ImageEnhance.Contrast(image)<br>
contrast = 1.5<br>
image_contrasted = enh_con.enhance(contrast)<br>
image_contrasted.show()<br>

#Sharpen<br>
enh_sha = ImageEnhance.Sharpness(image)<br>
sharpness = 1.5<br>
image_sharped = enh_sha.enhance(sharpness)<br>
image_sharped.show()<br>

OUTPUT:
![IMGenhance](https://user-images.githubusercontent.com/98145104/176426602-d4218919-580a-4356-9e1b-c344564be4ba.png)
![IMG_enhance](https://user-images.githubusercontent.com/98145104/176426616-b5458d09-721d-4ab7-b036-09cf8c8e2a15.png)

# 18.Program to morph an image.
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
from PIL import Image, ImageEnhance<br>
img = cv2.imread('tree.jpg',0)<br>
ax = plt.subplots(figsize=(20,10))<br>
kernel = np.ones((5,5), np.uint8)<br>
opening  = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)<br>
closing  = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)<br>
erosion  = cv2.erode(img,kernel,iterations = 1)<br>
dilation = cv2.dilate(img, kernel, iterations = 1)<br>
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)<br>
plt.subplot(151)<br>
plt.imshow(opening)<br>
plt.subplot(152)<br>
plt.imshow(closing)<br>
plt.subplot(153)<br>
plt.imshow(erosion)<br>
plt.subplot(154)<br>
plt.imshow(dilation)<br>
plt.subplot(155)<br>
plt.imshow(gradient)<br>
cv2.waitKey(0)<br>

OUTPUT:
![Enhance2](https://user-images.githubusercontent.com/98145104/176427041-69e01e67-d77a-4374-8330-e80cee6865ce.png)

# 19.Program to<br>(i)Read the image, convert it into grayscale image<br>(ii)write (save) the grayscale image and(iii)<br>
import cv2<br>
OriginalImg=cv2.imread('Chick.jpg')<br>
GrayImg=cv2.imread('Chick.jpg',0)<br>
isSaved=cv2.imwrite('D:/i.jpg', GrayImg)<br>
cv2.imshow('Display Original Image', OriginalImg)<br>
cv2.imshow('Display Grayscale Image',GrayImg)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
if isSaved:<br>
    print('The image is succrssfully saved')<br>
    
OUTPUT:
![chick_org](https://user-images.githubusercontent.com/98145104/178705096-61dce7f4-254f-472e-9bdd-42c696814328.png)
![chick_gray](https://user-images.githubusercontent.com/98145104/178705148-c0e05014-e757-4949-834a-ef14234dc08c.png)

# 20.Program to perform Graylevel slicing with background.
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
image=cv2.imread('man.jpg',0)<br>
x,y=image.shape<br>
z=np.zeros((x,y))<br>
for i in range(0,x): <br>
    for j in range(0,y):<br>
        if(image[i][j]>50 and image[i][j]<150):<br>
            z[i][j]=255<br>
        else:<br>
            z[i][j]=image[i][j]<br>
equ=np.hstack((image,z))<br>
plt.title('Graylevel slicing with background')<br>
plt.imshow(equ,'gray')<br>
plt.show()<br>

OUTPUT:
![graylevl_bg](https://user-images.githubusercontent.com/98145104/178705754-f6426bf8-96fd-4c94-974e-7539eb2b95a7.png)

# 21.Program to perform Graylevel slicing without background.
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
image=cv2.imread('man.jpg',0)<br>
x,y=image.shape<br>
z=np.zeros((x,y))<br>
for i in range(0,x):<br>
    for j in range(0,y):<br>
        if(image[i][j]>50 and image[i][j]<150):<br>
            z[i][j]=255<br>
        else:<br>
            z[i][j]=0<br>
equ=np.hstack((image,z))<br>
plt.title('Graylevel slicing without background')<br>
plt.imshow(equ,'gray')<br>
plt.show()<br>

OUTPUT:
![graylevl_wo_bg](https://user-images.githubusercontent.com/98145104/178706760-ae9fab0b-6a8d-4e61-ad37-3316f035ae5d.png)

# 22.Program to analyze the image data using Histogram.

//openCV<br>
import cv2<br>
import numpy as np<br>
img  = cv2.imread('man.jpg',0)<br>
hist = cv2.calcHist([img],[0],None,[256],[0,256])<br>
plt.hist(img.ravel(),256,[0,256])<br>
_ = plt.xlabel('Intensity Value')<br>
_ = plt.ylabel('Count') <br>
plt.show()<br>

OUTPUT:
![hist_openCV](https://user-images.githubusercontent.com/98145104/178967545-c87b0cb9-8890-4482-80e0-a51a4ec6e0a1.png)

//skimage<br>
from skimage import io<br>
import matplotlib.pyplot as plt<br>
image = io.imread('man.jpg')<br>
ax = plt.hist(image.ravel(), bins = 256)<br>
_ = plt.xlabel('Intensity Value')<br>
_ = plt.ylabel('Count') <br>
plt.show()<br>

OUTPUT:
![hist_skimage](https://user-images.githubusercontent.com/98145104/178970126-cb3e6432-2115-46d2-b358-814754f7a0fe.png)

# 23.Program to perform basic image data analysis using intensity transformation:
    a) Image negative
    b) Log transformation
    c) Gamma correction
    
%matplotlib inline<br>
import imageio<br>
import matplotlib.pyplot as plt<br>
#import warnings<br>
#import matplotlib.cbook<br>
#warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)<br>
pic=imageio.imread("violet.jpg")<br>
plt.figure(figsize=(6,6))<br>
plt.imshow(pic);<br>
plt.axis('off');<br>

OUTPUT:
![v_org](https://user-images.githubusercontent.com/98145104/179965458-81011338-3467-4462-bd14-f3fc3305ef4e.png)<br>

negative = 255 - pic <br>
plt.figure(figsize=(6,6))<br>
plt.imshow(negative);<br>
plt.axis('off');<br>

OUTPUT:
![v_neg](https://user-images.githubusercontent.com/98145104/179965536-f9c0992c-bba8-4b69-9442-81e3d0b46859.png)<br>

%matplotlib inline<br>

import cv2<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>

pic=cv2.imread('violet.jpg')<br>
gray=lambda rgb : np.dot(rgb[...,:3],[0.299,0.587,0.114])<br>
gray=gray(pic)<br>

max=np.max(gray)<br>

def logtransform():<br>
    return(255/np.log(1+max))*np.log(1+gray)<br>
plt.figure(figsize=(5,5))<br>
plt.imshow(logtransform(), cmap=plt.get_cmap(name='gray'))<br>
plt.axis('off');<br>

import imageio<br>
import matplotlib.pyplot as plt<br>

OUTPUT:
![v_log](https://user-images.githubusercontent.com/98145104/179965644-5bd8e2fd-f59c-4466-a670-35ae5379b5dc.png)<br>

# Gamma encoding<br>
pic=imageio.imread('violet.jpg')<br>
gamma=2.2 # Gamma < 1 = Dark; Gamma > 1 = Bright<br>

gamma_correction = ((pic/255)**(1/gamma))<br>
plt.figure(figsize=(5,5))<br>
plt.imshow(gamma_correction)<br>
plt.axis('off');<br>

OUTPUT:
![v_gamma](https://user-images.githubusercontent.com/98145104/179965672-6cde13f4-7c96-4b4a-b4fc-ed355bc86a53.png)<br>

# 24.Program to perform basic image manipulation:<br>
    a) Sharpness<br>
    b) Flipping<br>
    c) Cropping<br>
    
    #Image Sharpen<br>
from PIL import Image<br>
from PIL import ImageFilter<br>
import matplotlib.pyplot as plt<br>
#Load the image<br>
my_image=Image.open('dog1.jpg')<br>
#Use sharpen function<br>
sharp=my_image.filter(ImageFilter.SHARPEN)<br>
#Save the image<br>
sharp.save('E:/image_sharpen.jpg')<br>
sharp.show()<br>
plt.imshow(sharp)<br>
plt.show()<br>

OUTPUT:
![dog1](https://user-images.githubusercontent.com/98145104/179969841-ec422965-fc8f-4173-a943-a1654e6c5349.png)<br>

![E_dog](https://user-images.githubusercontent.com/98145104/179969933-4a0bff23-0c22-40e5-bfcb-93722d2674fe.png)<br>

#Image flip<br>
import matplotlib.pyplot as plt<br>
#Load the image<br>
img=Image.open('dog1.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
#use the flip function<br>
flip=img.transpose(Image.FLIP_LEFT_RIGHT)<br>
#save the image<br>
flip.save('E:/image_flip.jpg')<br>
plt.imshow(flip)<br>
plt.show()<br>

OUTPUT:
![E_dogg](https://user-images.githubusercontent.com/98145104/179970329-a24ccd04-452e-4bb1-a66f-9decded3766a.png)<br>
![dog_flip](https://user-images.githubusercontent.com/98145104/179970346-2439d12a-ab72-406f-b218-10fa19523fd9.png)<br>

Image crop<br>
#Importing Image class from <br>
from PIL import Image<br>
import matplotlib.pyplot as plt<br>
im=Image.open('dog1.jpg')<br>
width,height=im.size<br>
im1=im.crop((280,100,800,600))<br>
im1.show()<br>
plt.imshow(im1)<br>
plt.show()<br>

OUTPUT:<br>
![img_crop](https://user-images.githubusercontent.com/98145104/186378872-d9f0fb9c-b4f7-4a42-ac76-fe59a6f126f8.png)<br>
