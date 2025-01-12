import cv2
import numpy as np
import os
import mediapipe as mp

# Constants
brushThickness = 25
eraserThickness = 100

# Load header images
folderPath = "navbar"
myList = os.listdir(folderPath)
overlayList = [cv2.imread(f'{folderPath}/{imPath}') for imPath in myList]
print(f"Loaded {len(overlayList)} header images.")

# Initial settings
header = overlayList[0]
drawColor = (255, 0, 255)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# MediaPipe Hands setup
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.65)
mpDraw = mp.solutions.drawing_utils

# Canvas for drawing
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

# Variables for tracking
xp, yp = 0, 0

while True:
    # 1. Capture frame
    success, img = cap.read()
    if not success:
        print("Failed to capture frame.")
        break
    img = cv2.flip(img, 1)

    # 2. Detect hand landmarks
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    lmList = []

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))
            # Draw hand landmarks on the image
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # 3. If landmarks are detected, process gestures
    if lmList:
        # Tip of index and middle fingers
        x1, y1 = lmList[8][1:]  # Index finger tip
        x2, y2 = lmList[12][1:]  # Middle finger tip

        # 4. Check which fingers are up
        fingers = [0, 0, 0, 0, 0]
        if len(lmList) >= 21:  # Ensure all landmarks are available
            fingers[0] = 1 if lmList[4][1] < lmList[3][1] else 0  # Thumb
            fingers[1] = 1 if lmList[8][2] < lmList[6][2] else 0  # Index
            fingers[2] = 1 if lmList[12][2] < lmList[10][2] else 0  # Middle
            fingers[3] = 1 if lmList[16][2] < lmList[14][2] else 0  # Ring
            fingers[4] = 1 if lmList[20][2] < lmList[18][2] else 0  # Pinky

        # 5. Selection mode (two fingers up)
        if fingers[1] and fingers[2] and not fingers[3]:
            xp, yp = 0, 0
            print("Selection Mode")
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # 6. Drawing mode (only index finger up)
        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    # 7. Merge canvas and webcam feed
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # 8. Add header to the frame
    img[0:125, 0:1280] = header

    # 9. Display the result
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
