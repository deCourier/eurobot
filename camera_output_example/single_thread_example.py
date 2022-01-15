from picamera.array import PiRGBArray
from picamera import PiCamera
from time import time
from time import sleep
import cv2

num_frames = 0
current_time = time()
camera = PiCamera()
rawCapture = PiRGBArray(camera, size=(640, 480))
stream = camera.capture_continuous(rawCapture, format="bgr", use_video_port=True)

sleep(0.1)
for f in stream:
    frame = f.array
    
    cv2.imshow("Image", frame)
    key = cv2.waitKey(0) & 0xFF

    num_frames += 1
    if time() - current_time < 60:
        break

print(num_frames / (time() - current_time))