from PIL import Image
import face_recognition

image = face_recognition.load_image_file("biden.jpg")

face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")

print("I found {} face(s) in this photograph.".format(len(face_locations)))

for face_location in face_locations:

	top, right, bottom, left = face_location
	print("A face is located at pixel location Top: {}, left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

	# you can access the actual face itself like this:
	face_img = image[top:bottom, left:right]
	pil_image = Image.fromarray(face_image)
	pil_image.show()