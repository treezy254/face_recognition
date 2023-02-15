import os
from skimage.transform import resize
from skimage.feature import hog
import numpy as np
import cv2
import boto3
import json

def lambda_handler(event, context):

    # Instantiate S3 client
    s3 = boto3.client("s3")

    # Instantiate cascade for face detection
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface.xml')

    if event:

        # Read file from event (these are dummy variables to demonstrate the code)
        bucket = event["bucket"]
        key = event["key"]
        file_object = s3.get_object(Bucket=bucket, Key=key)
        file_content = file_object["Body"].read()

        # Convert bytes to 1-D array
        np_array = np.fromstring(file_content, np.uint8)

        # Read the buffer into a colour image
        image_np = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        # Converting colour image to grayscale
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Crop image around face
        for (x, y, w, h) in faces[:1]:
            face = image_np[y:y + h, x:x + w]
        
        # Resize image for HOG
        resized_img = resize(face, (128*4, 64*4))

        # Create HOG descriptor
        fd = hog(
            resized_img, 
            orientations=9, 
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), 
            visualize=False, 
            multichannel=True
        )

        return {
            'status_code': 200,
            'body': json.dumps(fd)
        }7