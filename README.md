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


7.Program to display the image attributes.
