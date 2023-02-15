import face_recognition
import cv2
from imutils import paths
import pickle
import os

imagePaths = list(paths.list_images('Images'))
foundEncodings = []
foundNames = []

for (i, imagePath) in enumerate(imagePaths):
	# Extract person's name from path
	name = imagePath.split(os.path.sep)[-2]
	# load the input image and convert it from BGR (Opencv) to dlib ordering (RGB)
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# Locate faces and save coordinates of a bounding box
	boxes = face_recognition.face_locations(rgb, model='hog')
	# Compute the facial encodings
	encodings = face_recognition.face_encodings(rgb, boxes,model='large')
	# loop over the encodings
	for encoding in encodings:
		foundEncodings.append(encoding)
		foundNames.append(name)


data = {"encodings": foundEncodings, "names": foundNames}
# use pickle to save the dictionary
f = open("face_enc", "wb")
f.write(pickle.dumps(data))
f.close