import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import os

# Open the camera
cap = cv2.VideoCapture(0)

# Check if camera is opened
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize Hand Detector
detector = HandDetector(maxHands=1)

# Ensure model files exist before initializing the classifier
model_path = "model/keras_model.h5"
labels_path = "model/labels.txt"

if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    exit()
if not os.path.exists(labels_path):
    print(f"Error: Labels file not found at {labels_path}")
    exit()

# Initialize Classifier
classifier = Classifier(model_path, labels_path)

offset = 20
imgSize = 300
labels = ["hello", "i love you", "no", "please", "thank you", "yes"]

# Define custom blue color
BLUE = (255, 147, 8)

while True:
    success, img = cap.read()
    if not success:
        print("Warning: Failed to capture frame. Retrying...")
        continue  

    hands, img = detector.findHands(img, draw=True)  # Keep white lines & dots on hand

    imgOutput = img.copy()

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure cropping is within bounds
        x1, y1 = max(0, x - offset), max(0, y - offset)
        x2, y2 = min(img.shape[1], x + w + offset), min(img.shape[0], y + h + offset)

        if x1 >= x2 or y1 >= y2:
            print("Warning: Invalid crop region. Skipping this frame...")
            continue

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            print("Warning: Empty crop. Skipping this frame...")
            continue

        aspectRatio = h / w

        try:
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wGap + wCal] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hGap + hCal, :] = imgResize
        except Exception as e:
            print(f"Error resizing image: {e}")
            continue

        # Get Prediction
        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        label_text = labels[index]  # Only display gesture name (no confidence %)

        # Draw updated bounding box
        cv2.rectangle(imgOutput, (x1, y1), (x2, y2), BLUE, 4, cv2.LINE_AA)

        # Draw label background
        cv2.rectangle(imgOutput, (x1, y1 - 50), (x1 + 200, y1), BLUE, cv2.FILLED)

        # Draw label text
        cv2.putText(imgOutput, label_text, (x1 + 10, y1 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

        # Ensure both images have the same height before stacking
        imgWhite_resized = cv2.resize(imgWhite, (300, 400))  # Match height of imgOutput
        combined = np.hstack([cv2.resize(imgOutput, (600, 400)), imgWhite_resized])

        cv2.imshow('Hand Recognition', combined)

    else:
        cv2.imshow('Hand Recognition', cv2.resize(imgOutput, (600, 400)))

    key = cv2.waitKey(1)
    if key == ord('q'):
        print("Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
