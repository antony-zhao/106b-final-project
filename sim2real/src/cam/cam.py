import cv2
import time
import torch
import numpy as np

# Open the default camera

# print(f"OpenCV version: {cv2.__version__}")

# max_cameras = 100
# avaiable = []
# for i in range(0, max_cameras):
#     cap = cv2.VideoCapture(i)
    
#     if not cap.read()[0]:
#         print(f"Camera index {i:02d} not found...")
#         continue
    
#     avaiable.append(i)
#     cap.release()
    
#     print(f"Camera index {i:02d} OK!")

# print(f"Cameras found: {avaiable}")

cam = cv2.VideoCapture(2)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cam.set(cv2.CAP_PROP_FPS, 30)

cap_times = []

while True:
    loop_start = time.time()
    ret, frame = cam.read()
    elapsed = time.time() - loop_start
    cap_times.append(elapsed)

    resized = cv2.resize(frame, (100, 100))
    grayscaled = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Display the captured frame
    cv2.imshow('Camera', grayscaled)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()
average = np.mean(cap_times)
print(average)