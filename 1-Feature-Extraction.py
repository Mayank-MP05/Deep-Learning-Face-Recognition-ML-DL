import numpy as np
import cv2
import os
import imutils
from imutils import paths
import pickle
import random 

training = 'training-images'
feature_op = 'features-saved'
detector = 'feature-extract-model'
feature_model = 'feature-extract-model/openface.t7'
least_confidence = 0.65

# We already have only facial part images still for the best
# results we are gonna use face detector on those face images

# Loading the Face Detection Caffe model
print("*** face detector started ... ***")
protoPath = os.path.sep.join([detector, "deploy.prototxt"])
modelPath = os.path.sep.join([detector, "face-area-detect.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
print("*** face detector done ... ***")

# Feature Extraction using torch model
print("*** Feature Extraction loading started ... ***")
embedder = cv2.dnn.readNetFromTorch(feature_model)
print("*** Feature Extraction loading  done ... ***")

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(training))
random.shuffle(imagePaths)
# print(imagePaths)

features_array = []
labels_array = []

total = 0

for (i, imagePath) in enumerate(imagePaths):
    name = imagePath.split(os.sep)[-2]
    print("[INFO] processing image {}/{} : {}".format(i + 1, len(imagePaths), name))

    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # Detecting the Face Area in the Image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # ensure at least one face was found
    if len(detections) > 0:

        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        # ensure that the detection with the largest probability also
        # means our minimum probability test (thus helping filter out
        # weak detections)
        if confidence > least_confidence:
            # compute the (x, y)-coordinates of the bounding box for
            # the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI and grab the ROI dimensions
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                             (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # add the name of the person + corresponding face
            # embedding to their respective lists
            labels_array.append(name)
            features_array.append(vec.flatten())
            total += 1

# print(total)
# print(features_array)
# print(labels_array)

# Saving With Pickle
data = {"featuers": features_array, "labels": labels_array}
f = open(f'{feature_op}/extracted-feature-data.pkl', "wb")
f.write(pickle.dumps(data))
f.close()
print('**** DATA SAVED SUCESS ****')


"""
(base) D:\LearnPythonQuarantine\face recognition\trip-DL>C:/Users/pc/Anaconda3/python.exe "d:/LearnPythonQuarantine/face recognition/trip-DL/1-Feature-Extraction.py"
*** face detector started ... ***
*** face detector done ... ***
*** Feature Extraction loading started ... ***
*** Feature Extraction loading  done ... ***
[INFO] quantifying faces...
[INFO] processing image 1/66
[INFO] processing image 2/66
[INFO] processing image 3/66
[INFO] processing image 4/66
[INFO] processing image 5/66
[INFO] processing image 6/66
[INFO] processing image 7/66
[INFO] processing image 8/66
[INFO] processing image 9/66
[INFO] processing image 10/66
[INFO] processing image 11/66
[INFO] processing image 12/66
[INFO] processing image 13/66
[INFO] processing image 14/66
[INFO] processing image 15/66
[INFO] processing image 16/66
[INFO] processing image 17/66
[INFO] processing image 18/66
[INFO] processing image 19/66
[INFO] processing image 20/66
[INFO] processing image 21/66
[INFO] processing image 22/66
[INFO] processing image 23/66
[INFO] processing image 24/66
[INFO] processing image 25/66
[INFO] processing image 26/66
[INFO] processing image 27/66
[INFO] processing image 28/66
[INFO] processing image 29/66
[INFO] processing image 30/66
[INFO] processing image 31/66
[INFO] processing image 32/66
[INFO] processing image 33/66
[INFO] processing image 34/66
[INFO] processing image 35/66
[INFO] processing image 36/66
[INFO] processing image 37/66
[INFO] processing image 38/66
[INFO] processing image 39/66
[INFO] processing image 40/66
[INFO] processing image 41/66
[INFO] processing image 42/66
[INFO] processing image 43/66
[INFO] processing image 44/66
[INFO] processing image 45/66
[INFO] processing image 46/66
[INFO] processing image 47/66
[INFO] processing image 48/66
[INFO] processing image 49/66
[INFO] processing image 50/66
[INFO] processing image 51/66
[INFO] processing image 52/66
[INFO] processing image 53/66
[INFO] processing image 54/66
[INFO] processing image 55/66
[INFO] processing image 56/66
[INFO] processing image 57/66
[INFO] processing image 58/66
[INFO] processing image 59/66
[INFO] processing image 60/66
[INFO] processing image 61/66
[INFO] processing image 62/66
[INFO] processing image 63/66
[INFO] processing image 64/66
[INFO] processing image 65/66
[INFO] processing image 66/66
**** DATA SAVED SUCESS ****
"""
