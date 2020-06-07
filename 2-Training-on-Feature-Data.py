from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle



# LoADING the saved data
data = pickle.load(open('features-saved/extracted-feature-data.pkl','rb'))
# print(data)

# here are two models
# 1. Face Detection Model used in Feature Extraction and Saving.py
# 2. Face Recognition used to run on top of feature extraction

le = LabelEncoder()
labels = le.fit_transform(data["labels"])

# Training the Model
print("**** training model... ****")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["featuers"], labels)
print("**** training model... ****")

# write the actual face recognition model to disk
f = open('face-recognition-op/recognition-svm-model.pkl', "wb")
f.write(pickle.dumps(recognizer))
f.close()
print("**** SAVED : Recognition Model *****")

# write the label encoder to disk
f = open("face-recognition-op/label-encoder.pkl", "wb")
f.write(pickle.dumps(le))
f.close()
print("**** SAVED : Label Encoder Model *****")

"""
(base) D:\LearnPythonQuarantine\face recognition\trip-DL>C:/Users/pc/Anaconda3/python.exe "d:/LearnPythonQuarantine/face recognition/trip-DL/2-Training-on-Feature-Data.py"
**** training model... ****
**** training model... ****
**** SAVED : Recognition Model *****
**** SAVED : Label Encoder Model *****
"""