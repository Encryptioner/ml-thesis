
'''
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

cv2.imshow("image", image)
cv2.waitKey(0)


# input in terminal $ python3 opencv_basic.py --image /home/ankur/Documents/player.png
or
# python3 opencv_basic.py --image /dataset/player.png
# which is the destination of data from the folder containing the .py file

'''


import cv2
import argparse

# Download the image used for this tutorial from here.
# http://goo.gl/jsYXl8

# Read the image
ap = argparse.ArgumentParser();
ap.add_argument("-i", "--image", required = True, help = "path to the image file");
args = vars(ap.parse_args());

# Read the image
image = cv2.imread(args["image"]);

# Convert the image into grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);

# Get the size of the image
# Output is in the form of tuple 
# (height of image , width of image, no of channels)
print(image.shape); # OUPTUT = (476, 640, 3)

# Accessing single pixel value
# image[y co-ordinate, x co-ordinate]
# Output is in the form of [B,G,R]
print(image[379, 383]); # OUPTUT = [207 151 143]

# To get value of only one channel, use
# image[y co-ordinate, x co-ordinate, channel index]
# Channel are indexed as -
# 0 for Blue, 1 for Green and 2 for Red
# In the example below we print(the Blue Channel Value
print(image[379, 383, 1]); # OUPTUT = 151

# Clone the image
image_copy = image.copy();

# Display the image
cv2.imshow("Image", image);
cv2.waitKey(); # The program will wait till eternity