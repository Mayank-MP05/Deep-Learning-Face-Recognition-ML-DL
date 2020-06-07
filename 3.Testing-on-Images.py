import cv2
import numpy as np
import pickle
import imutils
import os

training = 'training-images'
feature_op = 'features-saved'
detector = 'feature-extract-model'
feature_model = 'feature-extract-model/openface.t7'
least_confidence = 0.65
# importing Necessary Pickle Modules
# -----------------------------------------------------------------------
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

# Recognizer
recognizer = pickle.load(open('face-recognition-op/recognition-svm-model.pkl', "rb"))
le = pickle.load(open('face-recognition-op/label-encoder.pkl', "rb"))
# -----------------------------------------------------------------------

img = cv2.imread('testing-images/group-pic (3).jpg')
# cv2.imshow("Full Group Here ...",img)

image = imutils.resize(img, width=600)
(h, w) = image.shape[:2]

# construct a blob from the image
imageBlob = cv2.dnn.blobFromImage(
	cv2.resize(image, (300, 300)), 1.0, (300, 300),
	(104.0, 177.0, 123.0), swapRB=False, crop=False)

# Detcting Faces
detector.setInput(imageBlob)
detections = detector.forward()

for i in range(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the
	# prediction
	confidence = detections[0, 0, i, 2]

	# filter out weak detections
	if confidence > least_confidence:
		# compute the (x, y)-coordinates of the bounding box for the
		# face
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# extract the face ROI
		face = image[startY:endY, startX:endX]
		(fH, fW) = face.shape[:2]

		# ensure the face width and height are sufficiently large
		if fW < 20 or fH < 20:
			continue

		# construct a blob for the face ROI, then pass the blob
		# through our face embedding model to obtain the 128-d
		# quantification of the face
		faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
			(0, 0, 0), swapRB=True, crop=False)
		embedder.setInput(faceBlob)
		vec = embedder.forward()

		# perform classification to recognize the face
		preds = recognizer.predict_proba(vec)[0]
		j = np.argmax(preds)
		proba = preds[j]
		name = le.classes_[j]

		# draw the bounding box of the face along with the associated
		# probability
		text = "{}: {:.2f}%".format(name, proba * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		#cv2.rectangle(image, (startX, startY), (endX, endY),
		#	(0, 0, 255), 2)	
		cv2.circle(image, ((startX + endX)//2, (startY + endY)//2), (max((- startX + endX)//2, (- startY + endY)//2) + 10),
			(0, 255, 0), 2)		

		cv2.putText(image, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# show the output image
cv2.imshow("Drawn - Face Recognised ...", image)
cv2.waitKey(0)