# Image-Processing
SAMPLE-Progs
http://localhost:8889/tree/ImgProcessing_Sample%20-%20AllenPhilips

1.Program to display grayscale image using read and write operation.

import cv2 
img=cv2.imread('dog.jpg',0)
cv2.imshow('dog',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

OUTPUT:![Grayscale](https://user-images.githubusercontent.com/98145104/173812852-eb93e44b-8173-49e1-9d47-96ee02db6739.png)

2.Program to display the image using matplotlib.

import matplotlib.image as mping
import matplotlib.pyplot as plt
img = mping.imread('Car.jpg')
plt.imshow(img)

3.Program to perform linear transformation rotation.
import cv2
from PIL import Image
img=Image.open("vase.jpg")
img=img.rotate(180)
img.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

OUTPUT:![ImgRotate](https://user-images.githubusercontent.com/98145104/173812749-03040197-bc0e-44b9-a3ef-95ae5b41c649.png)

4.Program to convert color string to RGB color value.

5.Program to create image using colors.


6.Program to visualize the image using various color spaces.

7.Program to display the image attributes.
