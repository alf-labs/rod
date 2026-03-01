#!/usr/bin/python
# Ensure we're running from the Virtual Env version
import os
if not "VIRTUAL_ENV" in os.environ:
    print("ERROR: Run this from venv using 'source ./venv_catd/bin/activate' first")
    exit(1)
IS_RPI = os.path.isfile("/etc/rpi-issue")

import base64
import sys

try:
    import cv2
    import numpy as np
    import imutils
    from flask import Flask, render_template, Response, request, jsonify
except ModuleNotFoundError as e:
    print(f"ERROR: Missing library. {e}")
    print( "To fix: $ pip install opencv-python numpy imutils flask")
    print(f"or    : $ python {sys.argv[0]}")
    exit(1)

WINDOW_TITLE = "Rod Sample"
VIDEOS = [
    "../samples/rod1_front_randall_up_2025-03-23.mp4",
    "../samples/rod1_rear_randall_up_2025-03-23.mp4",
]

FPS = 30
FPS_MS = 1000//FPS

class Main:
    def __init__(self):
        pass

    def run(self):
        print("@@ Run")

        cap = cv2.VideoCapture(VIDEOS[0])
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imshow(WINDOW_TITLE, frame)
                if cv2.waitKey(FPS_MS) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

        print("@@ end")

if __name__ == "__main__":
    m = Main()
    m.run()

