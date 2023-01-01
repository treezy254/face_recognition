import face_recognition

image = face_recognition.load_image_file("your_file.jpg")
face_locations = face_recognition.face_locations(image)

# Get the locations and outlines of each person's eyes, nose, mouth and chin
face_landmarks_list = face_recognition.face_landmarks(image)

# Identify faces in pictures

import face_recognition

known_image = face_recognition.load_image_file("biden.jpg")
unknown_image = face_recognition.load_image_file("unknown.jpg")

biden_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([biden_encoding], unknown_encoding)