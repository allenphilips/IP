1)Python program to explain cv2.imshow() method.
import cv2
path='BUTTERFLY3.jpg'
i=cv2.imread(path,1)
cv2.imshow('image',i)
cv2.waitKey(0)
cv2.destroyAllWindows()

OUTPUT:
image

2)Develop a program to display grey scale image using read and write operations.
import cv2
img=cv2.imread('BUTTERFLY1.jpg',0)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

OUPUT:
image

3)Develop a program to display the image using matplotlib.
import matplotlib.image as mping
import matplotlib.pyplot as plt
img=mping.imread('FLOWER1.jpg')
plt.imshow(img)

OUTPUT:
download

4)Develop a program to perform linear transformation.
1-Rotation
2-Scalling
from PIL import Image
img=Image.open("LEAF1.jpg")
img=img.rotate(60)
img.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

OUTPUT:
image

5)Develop a program to convert color string to RGB color values.
from PIL import ImageColor
img1=ImageColor.getrgb("pink")
print(img1)
img2=ImageColor.getrgb("blue")
print(img2)

OUTPUT:
(255, 192, 203)
(0, 0, 255)

6)Write a program to create image using colors spaces.
from PIL import Image
img=Image.new('RGB',(200,400),(255,255,0))
img.show()

OUTPUT:
image

7)Develop a program to visualize the image using various color.
import cv2
import matplotlib.pyplot as plt
import numpy as np
img=cv2.imread('PLANT1.jpeg')
plt.imshow(img)
plt.show()
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
plt.imshow(img)
plt.show()

OUTPUT:
download
download
download

8)Write a program to display the image attributes.
from PIL import Image
image=Image.open('BUTTERFLY3.jpg')
print("Filename:",image.filename)
print("Format:",image.format)
print("Mode:",image.mode)
print("size:",image.size)
print("Width:",image.width)
print("Height:",image.height)
image.close()

OUTPUT:
Filename: BUTTERFLY3.jpg
Format: JPEG
Mode: RGB
size: (770, 662)
Width: 770
Height: 662

9)Resize the original image
import cv2
img=cv2.imread('FLOWER2.jpg')
print('Original image length width',img.shape)
cv2.imshow('Original image',img)
cv2.waitKey(0)
imgresize=cv2.resize(img,(150,160))
cv2.imshow('Resized image',imgresize)
print('Resized image length width',imgresize.shape)
cv2.waitKey(0)

OUTPUT:
image
image
Original image length width (668, 800, 3)
Resized image length width (160, 150, 3)

10)Convert the original image to gray scale and then to binary....
import cv2
img=cv2.imread('FLOWER3.jpeg')
cv2.imshow("RGB",img)
cv2.waitKey(0)
img=cv2.imread('FLOWER3.jpeg',0)
cv2.imshow("Gray",img)
cv2.waitKey(0)
ret,bw_img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
cv2.imshow("Binary",bw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

OUTPUT:
image
image
image

11)Develop a program to readimage using URL.
from skimage import io
import matplotlib.pyplot as plt
url='https://cdn.theatlantic.com/thumbor/viW9N1IQLbCrJ0HMtPRvXPXShkU=/0x131:2555x1568/976x549/media/img/mt/2017/06/shutterstock_319985324/original.jpg'
image=io.imread(url)
plt.imshow(image)
plt.show()

OUTPUT:
download

12)Write a program to mask and blur the image.
import cv2
import matplotlib.image as mping
import matplotlib.pyplot as plt
img=mping.imread('R.jpg')
plt.imshow(img)
plt.show()
download
hsv_img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
light_orange=(1, 190, 200)
dark_orange=(18, 255, 255)
mask=cv2.inRange(hsv_img,light_orange,dark_orange)
result=cv2.bitwise_and(img,img,mask=mask)
plt.subplot(1,2,1)
plt.imshow(mask,cmap="gray")
plt.subplot(1,2,2)
plt.imshow(result)
plt.show()
download
light_white=(0,0,200)
dark_white=(145,60,255)
mask_white=cv2.inRange(hsv_img,light_white,dark_white)
result_white=cv2.bitwise_and(img,img,mask=mask_white)
plt.subplot(1,2,1)
plt.imshow(mask_white,cmap="gray")
plt.subplot(1,2,2)
plt.imshow(result_white)
plt.show()
download
final_mask=mask+mask_white
final_result=cv2.bitwise_and(img,img,mask=final_mask)
plt.subplot(1,2,1)
plt.imshow(final_mask,cmap="gray")
plt.subplot(1,2,2)
plt.imshow(final_result)
plt.show()
download
blur=cv2.GaussianBlur(final_result, (7,7), 0)
plt.imshow(blur)
plt.show()
download

12.1(EXTRA)Write a program to mask and blur the image.
import cv2
import matplotlib.image as mping
import matplotlib.pyplot as plt
img=mping.imread('img.jpg')
plt.imshow(img)
plt.show()
download
light_white=(0,0,200)
dark_white=(145,60,255)
mask_white=cv2.inRange(hsv_img,light_white,dark_white)
result_white=cv2.bitwise_and(img,img,mask=mask_white)
plt.subplot(1,2,1)
plt.imshow(mask_white,cmap="gray")
plt.subplot(1,2,2)
plt.imshow(result_white)
plt.show()
download
blur=cv2.GaussianBlur(result_white, (7,7), 0)
plt.imshow(blur)
plt.show()
download

13)Write a program to perform arithmatic operations on images
import cv2
import matplotlib.image as mping
import matplotlib.pyplot as plt

#Reading image file
img1=cv2.imread('FLOWER1.jpg')
img2=cv2.imread('BUTTERFLY3.jpg')

#Applying numpy addition on images
fimg1 = img1 + img2
plt.imshow(fimg1)
plt.show()

#saving the output images
cv2.imwrite('output.jpg',fimg1)
fimg2 = img1 - img2
plt.imshow(fimg2)
plt.show()

#saving the output image
cv2.imwrite('output.jpg',fimg2)
fimg3 = img1 * img2
plt.imshow(fimg3)
plt.show()

#saving the output image
cv2.imwrite('output.jpg',fimg3)
fimg4 = img1 / img2
plt.imshow(fimg4)
plt.show()

#saving the output image
cv2.imwrite('output.jpg',fimg4)

OUTPUT:
download
download
download
download

14)Develop the program to change the image to different color spaces.
import cv2
img=cv2.imread("PLANT5.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
cv2.imshow("GRAY image",gray)
cv2.imshow("HSV image",hsv)
cv2.imshow("LAB image",lab)
cv2.imshow("HLS image",hls)
cv2.imshow("YUV image",yuv)
cv2.waitKey(0)
cv2.destroyAllWindows()

OUTPUT:
image
image
image
image
image

15)Program to create an image using 2D array
import cv2 as c
import numpy as np
from PIL import Image
array=np.zeros([100,200,3],dtype=np.uint8)
array[:,:100]=[255,130,0]
array[:,100:]=[0,0,255]
img=Image.fromarray(array)
img.save('IMAGES.jpg')
img.show()
c.waitKey(0)

OUTPUT:
image

16)Bitwise operation
import cv2
import matplotlib.pyplot as plt
image1=cv2.imread('BUTTERFLY1.png',1)
image2=cv2.imread('BUTTERFLY1.png')
ax=plt.subplots(figsize=(15,10))
bitwiseAnd=cv2.bitwise_and(image1,image2)
bitwiseOr=cv2.bitwise_or(image1,image2)
bitwiseXor=cv2.bitwise_xor(image1,image2)
bitwiseNot_img1=cv2.bitwise_not(image1)
bitwiseNot_img2=cv2.bitwise_not(image2)
plt.subplot(151)
plt.imshow(bitwiseAnd)
plt.subplot(152)
plt.imshow(bitwiseOr)
plt.subplot(153)
plt.imshow(bitwiseXor)
plt.subplot(154)
plt.imshow(bitwiseNot_img1)
plt.subplot(155)
plt.imshow(bitwiseNot_img2)
cv2.waitKey(0)

OUTPUT:
download

17)Blurring image
#importing libraries
import cv2
import numpy as np
image=cv2.imread('BUTTERFLY1.png')
cv2.imshow('Original Image',image)
cv2.waitKey(0)

#Gussian Blur
Gaussian=cv2.GaussianBlur(image,(7,7),0)
cv2.imshow('Gaussian Blurring',Gaussian)
cv2.waitKey(0)

#Median Blur
median=cv2.medianBlur(image,5)
cv2.imshow('Median Blurring',median)
cv2.waitKey(0)

#Bilateral Blur
bilateral=cv2.bilateralFilter(image,9,75,75)
cv2.imshow('Bilateral blurring',bilateral)
cv2.waitKey(0)
cv2.destroyAllWindows()

OUTPUT:
image
image
image
image

18)Image Enhancement
from PIL import Image
from PIL import ImageEnhance
image=Image.open('BUTTERFLY2.jpg')
image.show()
enh_bri=ImageEnhance.Brightness(image)
brightness=1.5
image_brightened=enh_bri.enhance(brightness)
image_brightened.show()
enh_col=ImageEnhance.Color(image)
color=1.5
image_colored=enh_col.enhance(color)
image_colored.show()
enh_con=ImageEnhance.Contrast(image)
contrast=1.5
image_contrasted=enh_con.enhance(contrast)
image_contrasted.show()
enh_sha=ImageEnhance.Sharpness(image)
sharpness=3.0
image_sharped=enh_sha.enhance(sharpness)
image_sharped.show()

OUTPUT:
image
image
image
image
image

19)Morpholigical operation
import cv2
import numpy as np
#from matplotlib import pyplt as plt
import matplotlib.pyplot as plt
from PIL import Image,ImageEnhance
img=cv2.imread('FLOWER1.JPG',0)
ax=plt.subplots(figsize=(20,10))
kernel=np.ones((5,5),np.uint8)
opening=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
closing=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
erosion=cv2.erode(img,kernel,iterations=1)
dilation=cv2.dilate(img,kernel,iterations=1)
gradient=cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)
plt.subplot(151)
plt.imshow(opening)
plt.subplot(152)
plt.imshow(closing)
plt.subplot(153)
plt.imshow(erosion)
plt.subplot(154)
plt.imshow(dilation)
plt.subplot(155)
plt.imshow(gradient)
cv2.waitKey(0)

OUTPUT:
download

20)Develop a program to
i)Read the image,convert it into grayscale image
ii)Write(save) the grayscale image and
iii)Display the original image and grayscale image
import cv2
OriginalImg=cv2.imread('FLOWER1.jpg')
GrayImg=cv2.imread('FLOWER1.jpg',0)
isSaved=cv2.imwrite('C:/thash/th.jpg',GrayImg)
cv2.imshow('Display Original Image',OriginalImg)
cv2.imshow('Display GrayScale Image',GrayImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
if isSaved:
print('The image is succesfully saved.')

OUTPUT:
image
The Image Is Successfully saved

21)Slicing with background
import cv2
import numpy as np
from matplotlib import pyplot as plt
image=cv2.imread('CAT1.jpg',0)
x,y=image.shape
z=np.zeros((x,y))
for i in range(0,x):
for j in range(0,y):
if(image[i][j]>50 and image[i][j]<150):
z[i][j]=255
else:
z[i][j]=image[i][j]
equ=np.hstack((image,z))
plt.title('Graylevel slicing with background')
plt.imshow(equ,'gray')
plt.show()

OUTPUT:
download

22)Slicing without background
import cv2
import numpy as np
from matplotlib import pyplot as plt
image=cv2.imread('CAT1.jpg',0)
x,y=image.shape
z=np.zeros((x,y))
for i in range(0,x):
for j in range(0,y):
if(image[i][j]>50 and image[i][j]<150):
z[i][j]=255
else:
z[i][j]=0
equ=np.hstack((image,z))
plt.title('Graylevel slicing without background')
plt.imshow(equ,'gray')
plt.show()

OUTPUT:
download

23)Analyze the image using Histogram
import numpy as np
import skimage.color
import skimage.io
import matplotlib.pyplot as plt

#read the image of a plant seedling as grayscale from the outset
image = skimage.io.imread(fname="DOG1.jpg",as_gray=True)
image1 = skimage.io.imread(fname="DOG1.jpg")

#display the image
fig, ax = plt.subplots()
plt.imshow(image, cmap="gray")
plt.show()

fig, ax = plt.subplots()
plt.imshow(image1,cmap="gray")
plt.show()

#create the histogram
histogram, bin_edges = np.histogram(image, bins=256, range=(0, 1))

#configure and draw the histogram figure
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixel count")
plt.xlim([0.0, 1.0]) # <- named arguments do not work here

plt.plot(bin_edges[0:-1], histogram) # <- or here
plt.show()

OUTPUT:
download
download
download

24)Program to perform basic image data analysis using intensity transformation:
a) Image negative
b) Log transformation
c) Gamma correction
#%matplotlib inline
import imageio
import matplotlib.pyplot as plt
#import warnings
#import matplotlib.cbook
#warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
pic=imageio.imread('PARROT1.jpg')
plt.figure(figsize=(6,6))
plt.imshow(pic);
plt.axis('off');

download

negative=255-pic #neg=(l-1)-img
plt.figure(figsize=(6,6))
plt.imshow(negative);
plt.axis('off');

download

%matplotlib inline

import imageio
import numpy as np
import matplotlib.pyplot as plt

pic=imageio.imread('PARROT1.jpg')
gray=lambda rgb:np.dot(rgb[...,:3],[0.299,0.587,0.114])
gray=gray(pic)

max_=np.max(gray)

def log_transform():
return(255/np.log(1+max_))*np.log(1+gray)
plt.figure(figsize=(5,5))
plt.imshow(log_transform(),cmap=plt.get_cmap(name='gray'))
plt.axis('off');

download

import imageio
import matplotlib.pyplot as plt

#Gamma encoding
pic=imageio.imread('PARROT1.jpg')
gamma=2.2 #Gamma<1~Dark;Gamma >~Bright

gamma_correction=((pic/255)**(1/gamma))
plt.figure(figsize=(5,5))
plt.imshow(gamma_correction)
plt.axis('off');

download

25)Program to perform basic image manipulation:
a) Sharpness
b) Flipping
c) Cropping
#Image sharpen
from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt

#Load the image
my_image=Image.open('FISH1.jpg')
plt.imshow(my_image)
plt.show()
#Use sharpen function
sharp=my_image.filter(ImageFilter.SHARPEN)
#Save the image
sharp.save('C:/thash/Image_sharpen.jpg')
sharp.show()
plt.imshow(sharp)
plt.show()

download
download
image

#Image flip
import matplotlib.pyplot as plt
#load the image
img=Image.open('FISH1.jpg')
plt.imshow(img)
plt.show()
#use the flip function
flip=img.transpose(Image.FLIP_LEFT_RIGHT)

#save the image
flip.save('C:/thash/Image_flip.jpg')
plt.imshow(flip)
plt.show()

download
download
image

#Image Crop

#Importing Image class from PIL module
from PIL import Image
import matplotlib.pyplot as plt
#Opens a image in RGB mode
im=Image.open('FISH1.jpg')

#Size of the image in pixels(size of original image)
#(This is not mandatory)
width,height=im.size

#Cropped image of above dimension
#(It will not Change original image)
im1=im.crop((50,200,3000,1600))

#Shows the image in image viewer
im1.show()
plt.imshow(im1)
plt.show()

download

#d) Roberts Edge Detection- Roberts cross operator #Roberts Edge Detection- Roberts cross operator import cv2 import numpy as np from scipy import ndimage from matplotlib import pyplot as plt roberts_cross_v=np.array([[1,0],[0,-1]]) roberts_cross_h=np.array([[0,1],[-1,0]])

img=cv2.imread("color.jpg",0).astype('float64') img/=255.0 vertical=ndimage.convolve(img,roberts_cross_v) horizontal=ndimage.convolve(img,roberts_cross_h)

edged_img=np.sqrt(np.square(horizontal)+np.square(vertical)) edged_img*=255 cv2.imwrite("Output.jpg",edged_img) cv2.imshow("OutputImage",edged_img) cv2.waitKey() cv2.destroyAllWindows()

from PIL import Image,ImageChops,ImageFilter
from matplotlib import pyplot as plt

#Create a PIL Image object
x=Image.open("x.png")
o=Image.open("o.png")

#Find out attributes of Image Objects
print('size of the image:',x.size, 'color mode:',x.mode)
print('size of the image:',o.size, 'color mode:',o.mode)

#plot 2 images one besides the other
plt.subplot(121),plt.imshow(x)
plt.axis('off')
plt.subplot(122),plt.imshow(o)
plt.axis('off')

#multiple images
merged=ImageChops.multiply(x,o)

#adding 2 images
add=ImageChops.add(x,o)

#convert colour mode
greyscale=merged.convert('L')
greyscale
size of the image: (256, 256) color mode: RGB
size of the image: (256, 256) color mode: RGB

OUTPUT:
download

#more attributes
image=merged

print('image size: ',image.size,
'\ncolor mode: ',image.mode,
'\nimage width: ',image.width,'| also represented by: ',image.size[0],
'\nimage height: ',image.height,'| also represented by: ',image.size[1],)

OUTPUT:
image size: (256, 256)
color mode: RGB
image width: 256 | also represented by: 256
image height: 256 | also represented by: 256

#mapping the pixels of the image so we can use them as coordinates
pixel=greyscale.load()

#a nested loop to parse through all the pixels in the image
for row in range(greyscale.size[0]):
for column in range(greyscale.size[1]):
if pixel[row,column]!=(255):
pixel[row,column]=(0)

greyscale
download

#1.invert image
invert=ImageChops.invert(greyscale)

#2.invert by substraction
bg=Image.new('L',(256,256),color=(255))#create a new image with a solid white background
subt=ImageChops.subtract(bg,greyscale)#substract image from background

#3.rotate
rotate=subt.rotate(45)
rotate
download

#gaussian blur
blur=greyscale.filter(ImageFilter.GaussianBlur(radius=1))

#edge detection
edge=blur.filter (ImageFilter.FIND_EDGES)
edge
download

#Change edge colours
edge=edge.convert('RGB')
bg_red=Image.new('RGB',(256,256),color=(255,0,0))

filled_edge=ImageChops.darker(bg_red,edge)
filled_edge
download
#save image in the directory
edge.save('processed.png')

Implement a program to perform various edge detection techniques
**a) Canny Edge detection **

#Canny Edge detection
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

loaded_image=cv2.imread("c7.jpg")
loaded_image=cv2.cvtColor(loaded_image,cv2.COLOR_BGR2RGB)

gray_image=cv2.cvtColor(loaded_image,cv2.COLOR_BGR2GRAY)

edged_image=cv2.Canny(gray_image,threshold1=30,threshold2=100)

plt.figure(figsize=(20,20))
plt.subplot(1,3,1)
plt.imshow(loaded_image,cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,3,2)
plt.imshow(gray_image,cmap="gray")
plt.axis("off")
plt.title("GrayScale Image")
plt.subplot(1,3,3)
plt.imshow(edged_image,cmap="gray")
plt.axis("off")
plt.title("Canny Edge Detected Image")
plt.show
OUTPUT: image image image

b) Edge detection schemas-the gradient(Sobel-first order derivatives)based edge detector and the Laplacian(2nd order derivative,so it is extremely sensitive to noise)based edge detector
#LapLacian and Sobel Edge detecting methods
import cv2
import numpy as np
from matplotlib import pyplot as plt

#Loading image
#img0=cv2.imread('sanFrancisco.jpg',)
img0=cv2.imread("c7.jpg")

#Converting to gray scale
gray=cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)

#remove noise
img=cv2.GaussianBlur(gray,(3,3),0)

#covolute with proper kernels
laplacian=cv2.Laplacian(img,cv2.CV_64F)
sobelx=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5) #X
sobely=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5) #y

plt.subplot(2,2,1),plt.imshow(img,cmap='gray')
plt.title('Original'),plt.xticks([]),plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap='gray')
plt.title('Laplacian'),plt.xticks([]),plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap='gray')
plt.title('Sobel X'),plt.xticks([]),plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap='gray')
plt.title('Sobel Y'),plt.xticks([]),plt.yticks([])

plt.show()
OUTPUT:
image
c) Edge detection using Prewitt Operator
#Edge detection using Prewitt operator
import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread('c7.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_gaussian=cv2.GaussianBlur(gray,(3,3),0)

#prewitt
kernelx=np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
img_prewittx=cv2.filter2D(img_gaussian,-1,kernelx)
img_prewitty=cv2.filter2D(img_gaussian,-1,kernely)

cv2.imshow("Original Image",img)
cv2.imshow("Prewitt X", img_prewittx)
cv2.imshow("Prewitt Y", img_prewitty)
cv2.imshow("Prewitt",img_prewittx+img_prewitty)
cv2.waitKey()
cv2.destroyAllWindows()

OUTPUT:
image image image image


d) Roberts Edge Detection-Roberts cross operator

#Roberts Edge Detection- Roberts cross operator
import cv2
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
roberts_cross_v=np.array([[1,0],[0,-1]])
roberts_cross_h=np.array([[0,1],[-1,0]])

img=cv2.imread("c7.jpg",0).astype('float64')
img/=255.0
vertical=ndimage.convolve(img,roberts_cross_v)
horizontal=ndimage.convolve(img,roberts_cross_h)

edged_img=np.sqrt(np.square(horizontal)+np.square(vertical))
edged_img*=255
cv2.imwrite("Output.jpg",edged_img)
cv2.imshow("OutputImage",edged_img)
cv2.waitKey()
cv2.destroyAllWindows()

OUTPUT:
image

import numpy as np
import cv2
import matplotlib.pyplot as plt

#Open the image
img = cv2.imread('dimage_damaged.png')
plt.imshow(img)
plt.show()

#load the mask
mask=cv2.imread('dimage_mask.png',0)
plt.imshow(mask)
plt.show()

#Inpaint
dst = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)

#Write the output.
cv2.imwrite('dimage_inpainted.png',dst)
plt.imshow(dst)
plt.show()

OUTPUT:
download
download
download

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['figure.figsize'] =(10,8)

def show_image(image,title='Image',cmap_type='gray'):
plt.imshow(image,cmap=cmap_type)
plt.title(title)
plt.axis('off')

def plot_comparison(img_original,img_filtered,img_title_filtered):
fig,(ax1,ax2)=plt.subplots(ncols=2, figsize=(10,8), sharex=True, sharey=True)
ax1.imshow(img_original,cmap=plt.cm.gray)
ax1.set_title('Original')
ax1.axis('off')
ax2.imshow(img_filtered,cmap=plt.cm.gray)
ax2.set_title('img_title_filtered')
ax2.axis('off')

from skimage.restoration import inpaint
from skimage.transform import resize
from skimage import color

image_with_logo=plt.imread('imlogo.png')

#Initialize the mask
mask=np.zeros(image_with_logo.shape[:-1])

#Set the pixels where the Logo is to 1
mask[210:272, 360:425] = 1

#Apply inpainting to remove the Logo
image_logo_removed = inpaint.inpaint_biharmonic(image_with_logo,
mask,
multichannel=True)

#show the original and Logo removed images
plot_comparison(image_with_logo,image_logo_removed,'Image with logo removed')

OUTPUT:
download

from skimage.util import random_noise

fruit_image=plt.imread('fruits.jpg')

#add noise to the image
noisy_image=random_noise(fruit_image)

#Show the original and resulting image
plot_comparison(fruit_image, noisy_image, 'Noisy image')

OUTPUT:
download

from skimage.restoration import denoise_tv_chambolle

noisy_image=plt.imread('noisy.jpg')

#Apply total variation filter denoising
denoised_image=denoise_tv_chambolle(noisy_image,multichannel=True)

#show the noisy and denopised image
plot_comparison(noisy_image,denoised_image,'Denoised Image')

OUTPUT:
download

from skimage.restoration import denoise_bilateral

landscape_image=plt.imread('noisy.jpg')

#Apply bilateral filter denoising
denoised_image=denoise_bilateral(landscape_image,multichannel=True)

#Show original and resulting images
plot_comparison(landscape_image, denoised_image,'Denoised Image')

OUTPUT:
download

from skimage.segmentation import slic
from skimage.color import label2rgb
import matplotlib.pyplot as plt
import numpy as np
face_image = plt.imread('face.jpg')
segments = slic(face_image, n_segments=400)
segmented_image=label2rgb(segments,face_image,kind='avg')
plt.imshow(face_image)
plt.show()
plt.imshow((segmented_image * 1).astype(np.uint8))
plt.show()

OUTPUT:
download
download

def show_image_contour(image,contours):
plt.figure()
for n, contour in enumerate(contours):
plt.plot(contour[:,1], contour[:,0],linewidth=3)
plt.imshow(image,interpolation='nearest',cmap='gray_r')
plt.title('Contours')
plt.axis('off')

from skimage import measure, data

#obtain the horse image
horse_image=data.horse()

#Find the contours with a constant level value of 0.8
contours=measure.find_contours(horse_image,level=0.8)

#Shows the image with contours found
show_image_contour(horse_image,contours)

OUTPUT:
download

from skimage.io import imread
from skimage.filters import threshold_otsu

image_dices=imread('diceimg.png')

#make the image grayscale
image_dices=color.rgb2gray(image_dices)

#Obtain the optimal thresh value
thresh=threshold_otsu(image_dices)

#Apply threshholding
binary=image_dices > thresh

#Find contours at aconstant value of 0.8
contours=measure.find_contours(binary,level=0.8)

#Show the image
show_image_contour(image_dices, contours)

OUTPUT:
download

#Create list with the shape of each contour
shape_contours=[cnt.shape[0] for cnt in contours]

#Set 50 as the maximum sixe of the dots shape
max_dots_shape=50

#Count dots in contours excluding bigger then dots size
dots_contours=[cnt for cnt in contours if np.shape(cnt)[0]<max_dots_shape]

#Shows all contours found
show_image_contour(binary, contours)

#Print the dice's number
print('Dices dots number:{}.'.format(len(dots_contours)))

OUTPUT:
Dices dots number:21.
download

28)a) Canny Edge detection
#Canny Edge detection
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

loaded_image=cv2.imread("color.jpg")
loaded_image=cv2.cvtColor(loaded_image,cv2.COLOR_BGR2RGB)

gray_image=cv2.cvtColor(loaded_image,cv2.COLOR_BGR2GRAY)

edged_image=cv2.Canny(gray_image,threshold1=30,threshold2=100)

plt.figure(figsize=(20,20))
plt.subplot(1,3,1)
plt.imshow(loaded_image,cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,3,2)
plt.imshow(gray_image,cmap="gray")
plt.axis("off")
plt.title("GrayScale Image")
plt.subplot(1,3,3)
plt.imshow(edged_image,cmap="gray")
plt.axis("off")
plt.title("Canny Edge Detected Image")
plt.show
download
#b) Edge detection schemes - the gradient (Sobel - first order derivatives)
#based edge detector and the Laplacian (2nd order derivative, so it is extremely sensitive to noise) based edge detector.
#LapLacian and Sobel Edge detecting methods
import cv2
import numpy as np
from matplotlib import pyplot as plt

#Loading image
#img0=cv2.imread('sanFrancisco.jpg',)
img0=cv2.imread("color.jpg")

#Converting to gray scale
gray=cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)

#remove noise
img=cv2.GaussianBlur(gray,(3,3),0)

#covolute with proper kernels
laplacian=cv2.Laplacian(img,cv2.CV_64F)
sobelx=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5) #X
sobely=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5) #y

plt.subplot(2,2,1),plt.imshow(img,cmap='gray')
plt.title('Original'),plt.xticks([]),plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap='gray')
plt.title('Laplacian'),plt.xticks([]),plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap='gray')
plt.title('Sobel X'),plt.xticks([]),plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap='gray')
plt.title('Sobel Y'),plt.xticks([]),plt.yticks([])

plt.show()
download
#c) Edge detection using Prewitt Operator
#Edge detection using Prewitt operator
import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread('color.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_gaussian=cv2.GaussianBlur(gray,(3,3),0)

#prewitt
kernelx=np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
img_prewittx=cv2.filter2D(img_gaussian,-1,kernelx)
img_prewitty=cv2.filter2D(img_gaussian,-1,kernely)

cv2.imshow("Original Image",img)
cv2.imshow("Prewitt X", img_prewittx)
cv2.imshow("Prewitt Y", img_prewitty)
cv2.imshow("Prewitt",img_prewittx+img_prewitty)
cv2.waitKey()
cv2.destroyAllWindows()

#d) Roberts Edge Detection- Roberts cross operator
#Roberts Edge Detection- Roberts cross operator
import cv2
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
roberts_cross_v=np.array([[1,0],[0,-1]])
roberts_cross_h=np.array([[0,1],[-1,0]])

img=cv2.imread("color.jpg",0).astype('float64')
img/=255.0 vertical=ndimage.convolve(img,roberts_cross_v)
horizontal=ndimage.convolve(img,roberts_cross_h)

edged_img=np.sqrt(np.square(horizontal)+np.square(vertical))
edged_img*=255
cv2.imwrite("Output.jpg",edged_img)
cv2.imshow("OutputImage",edged_img)
cv2.waitKey()
cv2.destroyAllWindows()
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

OUTPUT:
![v_log](https://user-images.githubusercontent.com/98145104/179965644-5bd8e2fd-f59c-4466-a670-35ae5379b5dc.png)<br>

# Gamma encoding<br>
import imageio<br>
import matplotlib.pyplot as plt<br>
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
<br>
# 25.Program to perform edge detection:<br>
#Canny Edge detection<br>
import cv2<br>
import numpy as np <br>
import matplotlib.pyplot as plt<br>

plt.style.use('seaborn')<br>
<br>
loaded_image = cv2.imread("mario.jpg")<br>
loaded_image = cv2.cvtColor(loaded_image,cv2.COLOR_BGR2RGB)<br>
<br>
gray_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2GRAY)<br>
edged_image = cv2.Canny(gray_image, threshold1=30, threshold2=100)<br>
<br>
plt.figure(figsize=(20,20))<br>
plt.subplot(1,3,1)<br>
plt.imshow(loaded_image, cmap="gray") <br>
plt.title("Original Image")<br>
plt.axis("off")<br>
plt.subplot(1,3,2)<br>
plt.imshow(gray_image, cmap="gray")<br>
plt.axis("off")<br>
plt.title("GrayScale Image")<br>
plt.subplot(1,3,3)<br>
plt.imshow(edged_image,cmap="gray")<br>
plt.axis("off")<br>
plt.title("Canny Edge Detected Image")<br>
plt.show()<br>
<br>
OUTPUT:<br>
![Screenshot 2022-09-01 163313](https://user-images.githubusercontent.com/98145104/187899338-99da74af-4112-420c-a1d9-7f38ce379081.png)<br>
<br>
#Laplacian and Sobel Edge detecting methods<br>
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
<br>
#Loading image<br>
#img0 = cv2.imread('SanFrancisco.jpg',) <br>
img0 = cv2.imread('mario.jpg',)<br>
<br>
#converting to gray scale<br>
gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)<br>
<br>
#remove noise<br>
img= cv2.GaussianBlur(gray,(3,3),0)<br>
<br>
#convolute with proper kernels<br>
laplacian= cv2.Laplacian(img,cv2.CV_64F) <br>
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5) #x <br>
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5) #y<br>
<br>
plt.subplot(2,2,1), plt.imshow(img,cmap = 'gray')<br>
plt.title('Original'), plt.xticks([]), plt.yticks([]) <br>
plt.subplot(2,2,2), plt.imshow(laplacian, cmap = 'gray')<br>
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])<br>
plt.subplot(2,2,3), plt.imshow(sobelx,cmap = 'gray') <br>
plt.title('Sobel x'), plt.xticks([]), plt.yticks([])<br>
plt.subplot(2,2,4), plt.imshow(sobely,cmap = 'gray') <br>
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])<br>
plt.show()<br>
<br>
OUTPUT:<br>
![Screenshot 2022-09-01 163340](https://user-images.githubusercontent.com/98145104/187899447-03b04e6b-2ddc-4872-aad3-46d9abb209e1.png)<br>
<br>
#Edge detection using Prewitt operator<br>
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
<br>
img = cv2.imread('mario.jpg')<br>
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) <br>
img_gaussian = cv2.GaussianBlur (gray, (3,3),0)<br>
<br>
#prewitt<br>
kernelx = np.array([[1,1,1], [0,0,0],[-1,-1,-1]])<br>
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])<br>
img_prewittx = cv2.filter2D (img_gaussian, -1, kernelx) <br>
img_prewitty = cv2.filter2D (img_gaussian, -1, kernely)<br>
cv2.imshow("Original Image", img)<br>
cv2.imshow("Prewitt x", img_prewittx)<br>
cv2.imshow("Prewitt y", img_prewitty)<br>
cv2.imshow("Prewitt", img_prewittx + img_prewitty)<br>
cv2.waitKey()<br>
cv2.destroyAllwindows()<br>
<br>
OUTPUT:<br>
![Screenshot 2022-09-01 162418](https://user-images.githubusercontent.com/98145104/187899527-9b252b35-7075-4864-a404-a3277ccf6bd7.png)<br>
<br>
#Roberts Edge Detection- Roberts cross operator<br>
import cv2<br>
import numpy as np<br>
from scipy import ndimage<br>
from matplotlib import pyplot as plt <br>
<br>
roberts_cross_v = np.array([[1, 0 ],[0,-1 ]] )<br>
roberts_cross_h = np.array([[ 0, 1 ], [-1, 0 ]])<br>

img= cv2.imread("mario.jpg",0).astype('float64')<br>
img/=255.0<br>
vertical=ndimage.convolve( img, roberts_cross_v ) <br>
horizontal=ndimage.convolve( img, roberts_cross_h)<br>
   <br>                                   
edged_img = np.sqrt(np.square(horizontal) + np.square(vertical))<br>
edged_img*=255<br>
cv2.imwrite("output.jpg", edged_img)<br>
cv2.imshow("OutputImage", edged_img)<br>
cv2.waitKey()<br>
cv2.destroyAllwindows()<br>

OUTPUT:<br>
![Screenshot 2022-09-01 163123](https://user-images.githubusercontent.com/98145104/187899603-ba95dd51-b9dc-468a-a32c-9469da88ac99.png)<br>
