import cv2

for index in range(5):
    cap = cv2.VideoCapture(index)
    if cap.read()[0]:
        print("Camera found at index", index)
        cap.release()
    else:
        print("No camera at index", index)
