#importing all the liberaries
import cv2
import os
import numpy as np
from PIL import Image 

#Create LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Create face detector
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):# for images path
    
    # Get the path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # Create empty lists to store faces and IDs
    faceSamples = []
    Ids = []
    # Loop through all the image paths and load the IDs and the images
    for imagePath in imagePaths:
        # Load the image and convert it to grayscale
        pilImage = Image.open(imagePath).convert('L')
        # Convert the PIL image into a numpy array
        imageNp = np.array(pilImage, 'uint8')
        # Get the ID from the image path
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        # Detect faces in the image
        faces = detector.detectMultiScale(imageNp)
        # Loop through each detected face
        for (x, y, w, h) in faces:
            # Extract the face region from the image
            faceSamples.append(imageNp[y:y+h, x:x+w])
            Ids.append(Id)
    return faceSamples, Ids

# Get faces and IDs from the training image directory
faces, Ids = getImagesAndLabels('TrainingImage')

# Ensure that there are samples to train on
if len(faces) > 0:
    # Train the recognizer with the faces and IDs
    recognizer.train(faces, np.array(Ids))
    # Save the trained model to a file
    recognizer.save('TrainingImageLabel/trainner.yml')
    print("Training completed successfully.")
else:
    print("No training data found. Please provide samples to train the model.")
