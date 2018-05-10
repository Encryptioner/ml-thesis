
# import the necessary packages
from lbp1_localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC, SVC, NuSVC
from imutils import paths
import argparse
import cv2
from time import time
import pickle
import numpy as np
import matplotlib.pyplot as plt


# import inspect
# print(inspect.getsource(pickle))  # firs  import the module then

start_time=time()


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True,
                help="path to the training images")

args = vars(ap.parse_args())

if args == None:
    raise Exception("could not load image !")


# initialize the local binary patterns descriptor along with
# the data and label lists


radius = 8
numPoints = 24  # 8*radius

desc = LocalBinaryPatterns(numPoints, radius)  # (24,8)

data = []
labels = []
before_training=time()
print("Training is starting:\n")


# loop over the training images
for imagePath in paths.list_images(args["training"]):
    # load the image, convert it to grayscale, and describe it

    print("path = ", imagePath)

    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)

    #

    # extract the label from the image path, then update the
    # label and data lists
    labels.append(imagePath.split("/")[-2])
    data.append(hist)


# c1, ran1, tol1, w1, ver1,  iter1, pen, los, dua, mul, fit, sca
# c2, ran2, ker, gam , tol2, w2, ver2, , iter2,  deg, co, pro, shr, cac

classifier, c, ran, ker, gam, it = (LinearSVC, 500.0, None, 'linear', 0.0, 1000)


# train a SVM on the data

model = classifier(C=c, random_state=ran, max_iter=it)
# model = classifier(C=c, random_state=ran, kernel=ker, gamma=gam, max_iter=it)


# model = LinearSVC(penalty='12', loss='squared_hinge', dual=True, tol=1e-4, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)

# model = SVC(C=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, tol=1e-3, shrinking=True, probability=False,tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)

model.fit(data, labels)


# print trainig time and show data and labels
training_labels=labels
after_training=time()
training_time=after_training-before_training

# print("data = \n",data)
# print("training labels = \n",training_labels)
print("total trained data = ", len(labels))
print("training time = ", training_time, " seconds\n")


# saving data in file for split training & testing to different python script

f = open('dataset/trained_ck+.pckl', 'wb')
# f = open('dataset/trained_jaffe.pckl', 'wb')
# f = open('dataset/trained_ck+jaffe.pckl', 'wb')
# f = open('dataset/trained_object.pckl', 'wb')

pickle.dump([data, model, labels, desc, imagePath, training_time, numPoints, classifier, c, ran, ker, gam, it], f)
f.close()

#

# 0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise

# for classifier same type of image should be in dedicated folder

# ck+ (trained using ck+ - in expression folder)
# python3 lbp1_training.py --training dataset/expression/ck+

# jaffe (trained using jaffe - in expression folder)
# python3 lbp1_training.py --training dataset/expression/jaffe

# ck+jaffe (trained using ck+jaffe - in expression folder)
# python3 lbp1_training.py --training dataset/expression/ck+jaffe

# object
# python3 lbp1_training.py --training dataset/object/train

