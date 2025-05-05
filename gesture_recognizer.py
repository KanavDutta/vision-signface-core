import cv2
import time
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# ---------- CONFIG ----------
WINDOW_NAME = "Gesture Recognizer"
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480
WELCOME_DURATION = 2  # seconds

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_COLOR = (200, 255, 200)
THICKNESS = 1
LINE_TYPE = cv2.LINE_AA
# ----------------------------

# ----- Welcome Screen -----
welcome_img = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)  # Jet black background

# Centered text positions
main_text = "Gesture Recognizer"
sub_text = "Made by Kanav Dutta"

(main_w, _), _ = cv2.getTextSize(main_text, FONT, 1.5, 2)
(sub_w, _), _ = cv2.getTextSize(sub_text, FONT, 0.6, 1)

main_x = (WINDOW_WIDTH - main_w) // 2
main_y = WINDOW_HEIGHT // 2 - 20
sub_x = (WINDOW_WIDTH - sub_w) // 2
sub_y = main_y + 40

cv2.putText(welcome_img, main_text, (main_x, main_y),
            FONT, 1.5, (255, 255, 255), 2, LINE_TYPE)

cv2.putText(welcome_img, sub_text, (sub_x, sub_y),
            FONT, 0.6, (180, 180, 180), 1, LINE_TYPE)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.imshow(WINDOW_NAME, welcome_img)
cv2.waitKey(WELCOME_DURATION * 1000)

# ----- Webcam & Detector Setup -----
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)

# ----- Main Loop -----
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
    
    hands, frame = detector.findHands(frame)

    gesture = "No hand"
    if hands:
        hand = hands[0]
        fingers = detector.fingersUp(hand)

        if fingers == [0, 0, 0, 0, 0]:
            gesture = "Fist"
        elif fingers == [1, 1, 1, 1, 1]:
            gesture = "Open"
        elif fingers == [1, 0, 0, 0, 0]:
            gesture = "Thumbs Up"
        elif fingers == [0, 1, 1, 0, 0]:
            gesture = "Peace"
        else:
            gesture = f"{sum(fingers)} fingers"

    # Minimal overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (220, 55), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

    # Display gesture text
    cv2.putText(frame, gesture, (25, 45),
                FONT, FONT_SCALE, FONT_COLOR, THICKNESS, LINE_TYPE)

    cv2.imshow(WINDOW_NAME, frame)

    # Exit if 'q' pressed or window closed
    if (cv2.waitKey(1) & 0xFF == ord('q') or
        cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1):
        break

cap.release()
cv2.destroyAllWindows()
