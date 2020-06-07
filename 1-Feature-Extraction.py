import numpy as np
import cv2
import os

training = 'training-images'
feature_op = 'featuress-saved'
detector = 'feature-extract-model'
feature_model = 'feature-extract-model/openface.t7'

# We already have only facial part images still for the best
# results we are gonna use face detector on those face images

# Loading the Face Detection Caffe model
print("*** face detector started ... ***")
protoPath = os.path.sep.join([detector, "deploy.prototxt"])
modelPath = os.path.sep.join([detector,"face-area-detect.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
print("*** face detector done ... ***")

# Feature Extraction using torch model
print("*** Feature Extraction loading started ... ***")
embedder = cv2.dnn.readNetFromTorch(feature_model)
print("*** Feature Extraction loading  done ... ***")

