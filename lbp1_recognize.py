# https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/


# import the necessary packages
from lbp1_localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import cv2
from time import time
import pickle


start_time=time()


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True,
                help="path to the training images")
ap.add_argument("-e", "--testing", required=True,
                help="path to the tesitng images")
args = vars(ap.parse_args())

if args == None:
    raise Exception("could not load image !")


# initialize the local binary patterns descriptor along with
# the data and label lists
desc = LocalBinaryPatterns(24, 8)
data = []
labels = []
before_training=time()
print("Training is starting:\n")


# loop over the training images
for imagePath in paths.list_images(args["training"]):
    # load the image, convert it to grayscale, and describe it
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)

    # extract the label from the image path, then update the
    # label and data lists
    labels.append(imagePath.split("/")[-2])
    data.append(hist)


# train a Linear SVM on the data
model = LinearSVC(C=100.0, random_state=42)
model.fit(data, labels)


# print trainig time and show data and labels
training_labels=labels
after_training=time()
training_time=after_training-before_training
#print("data = \n",data)
#print("training labels = \n",training_labels)
print("total trained data = ",len(labels))
print("training time = ", training_time, " seconds\n")


# saving data in file for split training & testing to different python script

#f = open('dataset/trained_ck+0.pckl', 'wb')
#f = open('dataset/trained_ck+1.pckl', 'wb')
#f = open('dataset/trained_object.pckl', 'wb')

#pickle.dump([model, labels, desc, imagePath, training_time], f)
#f.close()

#############################

print("Testing is staring:\n")

true=0
false=0
labels.clear()


# loop over the testing images
for imagePath in paths.list_images(args["testing"]):
    # load the image, convert it to grayscale, describe it,
    # and classify it
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    hist = hist.reshape(1, -1)
    prediction = model.predict(hist)[0]


    # append the labels of testing folder of given dataset in cleared label array
    labels.append(imagePath.split("/")[-2])
    print("actually = ", labels[-1])
    print("predicted = ", prediction)
    print("path = ", imagePath)

    if labels[-1]==prediction:
        true=true+1
    else:
        false=false+1


    # display the image and the prediction
    cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255), 3)
    cv2.imshow("Image", image)
    cv2.waitKey(10)

#

end_time=time()
testing_labels=labels
#print("\ntesting labels = \n",testing_labels)
print("total tested data = ", true+false)

print("correctly predicted = ", true)
print("wrongly predicted = ", false)
print("testing accuracy", (true/(true+false))*100, "%")
print("training time = ", training_time, " seconds\n")


#

# 0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise

# for classifier same type of image should be in dedicated folder
# ck+0
# python3 lbp1_recognize.py --training dataset/ck+/training --testing dataset/ck+/testing
# ck+1
# python3 lbp1_recognize.py --training dataset/ck+/testing --testing dataset/ck+/training
# object
# python3 lbp1_recognize.py --training dataset/object/training --testing dataset/object/testing

