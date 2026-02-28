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
    from ultralytics import YOLO
except ModuleNotFoundError as e:
    print(f"ERROR: Missing library. {e}")
    print( "To fix: $ pip install opencv-python-headless numpy imutils flask ultralytics")
    print(f"or    : $ python {sys.argv[0]}")
    exit(1)



class Main:
    def __init__(self):
        pass

    def run(self):
        print("@@ Run")


if __name__ == "__main__":
    m = Main()
    m.run()

