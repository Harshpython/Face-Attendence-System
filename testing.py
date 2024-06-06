import cv2
import numpy as np

# Load the trained LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('TrainingImageLabel/trainner.yml')

# Load the Haar cascade classifier for face detection
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# Define the font for displaying text on the image
font = cv2.FONT_HERSHEY_SIMPLEX

# Start capturing frames from the camera
cam = cv2.VideoCapture(0)
while True:
    # Read a frame from the camera
    ret, im = cam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)

    # Loop through each detected face
    for (x, y, w, h) in faces:
        # Recognize the face and get the ID and confidence level
        Id, conf = recognizer.predict(gray[y:y+h, x:x+w])

        # Draw a rectangle around the detected face
        cv2.rectangle(im, (x, y), (x + w, y + h), (6, 260, 0), 7)

        # Display the ID of the recognized face on the image
        cv2.putText(im, str(Id), (x, y-40), font, 2, (255, 255, 255), 3)

    # Display the processed frame
    cv2.imshow('im', im)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
