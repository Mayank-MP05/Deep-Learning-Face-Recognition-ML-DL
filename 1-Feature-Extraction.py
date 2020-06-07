import numpy as np
import cv2
import os
import imutils
from imutils import paths

training = 'training-images'
feature_op = 'featuress-saved'
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
# print(imagePaths)

features_array = []
labels_array = []

total = 0

for (i, imagePath) in enumerate(imagePaths):
    print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
    name = imagePath.split(os.sep)[-1]

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

print(total)
print(features_array)
print(labels_array)