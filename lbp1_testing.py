
# import the necessary packages
from imutils import paths
import argparse
import cv2
from time import time
import pickle
#import warnings
#warnings.filterwarnings("ignore")  # ignores warning if occured

# load trained data

f = open('dataset/trained_ck+.pckl', 'rb')
# f = open('dataset/trained_jaffe.pckl', 'rb')
# f = open('dataset/trained_ck+jaffe.pckl', 'rb')
# f = open('dataset/trained_object.pckl', 'rb')

data, model, labels, desc, imagePath, training_time, numPoints, classifier, c, ran, ker, gam, it = pickle.load(f)
f.close()


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--testing", required=True,
                help="path to the tesitng images")
args = vars(ap.parse_args())

if args == None:
    raise Exception("could not load image !")


print("Testing is staring:\n")

true=0
false=0
labels.clear()
prediction=0


# loop over the testing images
for imagePath in paths.list_images(args["testing"]):
    # load the image, convert it to grayscale, describe it,
    # and classify it
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist = desc.describe(gray)
    hist = hist.reshape(1, -1)  # for not reshaping here

    prediction = model.predict(hist)[0]  # DeprecationWarning is showed for this line without reshape

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
    cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    cv2.imshow("Image", image)
    cv2.waitKey(10)

#

end_time=time()
testing_labels = labels
#print("\ntesting labels = \n",testing_labels)
print("total tested data = ", true+false)
print("correctly predicted = ", true)
print("wrongly predicted = ", false)
print("training time = ", training_time, 'seconds\n')
print("testing accuracy", (true/(true+false))*100, "%\n")



#

# 0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise

# for classifier same type of image should be in dedicated folder

# ck+ (trained using ck+ - in expression folder)
# python3 lbp1_testing.py --testing dataset/expression/ck+

# jaffe (trained using jaffe - in expression folder)
# python3 lbp1_testing.py --testing dataset/expression/jaffe

# ck+jaffe (trained using ck+jaffe - in expression folder)
# python3 lbp1_testing.py --testing dataset/expression/ck+jaffe

# object
# python3 lbp1_testing.py --testing dataset/object/test  or  python3 lbp1_testing.py --testing dataset/object/train


