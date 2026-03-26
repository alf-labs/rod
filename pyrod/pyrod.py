#!/usr/bin/python
# Ensure we're running from the Virtual Env version
import os
if not "VIRTUAL_ENV" in os.environ:
    print("ERROR: Run this from venv using 'source ./venv_catd/bin/activate' first")
    exit(1)
IS_RPI = os.path.isfile("/etc/rpi-issue")

import argparse
import base64
import json
import sys
import time

try:
    import cv2
    import numpy as np
    import imutils
    import scipy
    from flask import Flask, render_template, Response, request, jsonify
    from process_locator import Locator
except ModuleNotFoundError as e:
    print(f"ERROR: Missing library. {e}")
    print( "To fix: $ pip install opencv-python numpy scipy imutils flask")
    print(f"or    : $ python {sys.argv[0]}")
    exit(1)

WINDOW_TITLE = "Rod Sample"
IN_VIDEOS = [
    "../samples/rod1_front_randall_up_2025-03-23.mp4",
    "../samples/rod1_rear_randall_up_2025-03-23.mp4",
]

OUT_VIDEO_FILE_PATH = "outputIDX_%s.mp4" % time.strftime("%Y-%m-%d_%H-%M-%S")


class Main:
    def __init__(self):
        self.mx = 0
        self.my = 0
        self.zoom = 1
        self.skip_num = 1

    def print_fps(self, loop_s, dest):
        fps = 1/loop_s if loop_s > 0 else 0
        ms = int(loop_s * 1000)
        text = f"{self.mx:03d} x, {ms} ms, {fps:.2f} fps"
        z = self.zoom
        cv2.putText(dest, text,
            (10 * z, 30 * z),           # bottom-left coord
            cv2.FONT_HERSHEY_DUPLEX,    # font
            z,                          # font scale
            (0, 255, 255),              # color
            z )                         # line thickness

    def run(self):
        print("@@ Run")

        parser = argparse.ArgumentParser(description="PyRod")
        parser.add_argument("-i", "--input", default="0", help="Input video")
        parser.add_argument("-o", "--output", default=OUT_VIDEO_FILE_PATH, help="Output video")
        parser.add_argument("-n", "--no-video", action="store_true", help="Skip Video Output")
        args = parser.parse_args()

        input_idx = "_"
        input_path = args.input
        if input_path.isdigit():
            input_idx = int(input_path)
            input_path = IN_VIDEOS[input_idx % len(IN_VIDEOS)]
        output_path = f"{args.output}".replace("IDX", str(input_idx))
        detector_path = output_path.replace(".mp4", "") + ".1.json"
        print("Input:", input_path)
        print("Output:", output_path, "(disabled by -n)" if args.no_video else "")

        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TITLE, 1920//2, 1080//2)
        def _mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_MOUSEMOVE:
                self.mx = x
                self.my = y
        cv2.setMouseCallback(WINDOW_TITLE, _mouse_callback)

        processor = Locator()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        cap = cv2.VideoCapture(input_path)
        writer = None
        try:
            # Get input video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps_ms = int(1000 / fps)
            loop_s = 0
            init_once = True
            frame_count = 0
            paused = False
            view_org = True

            if args.no_video == False:
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
                print(f"@@ Writing {width}x{height}@{fps} fps to", output_path)

            last_frame = None
            while cap.isOpened():
                start_loop_s = time.perf_counter()
                if paused:
                    frame = last_frame.copy()
                else:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                    last_frame = frame.copy()
                    if view_org and frame_count == 50:
                        view_org = False

                _skip_num = self.skip_num
                if _skip_num > 1:
                    if frame_count % _skip_num != 0:
                        continue

                if init_once:
                    print(f"Video size: {width}x{height}")
                    processor.init_size(width, height)
                    init_once = False

                processor.init_overlay(frame)
                result = processor.filter(frame_count, frame)

                self.print_fps(loop_s, processor.overlay)

                if view_org:
                    show_frame = processor.combine_overlay(frame)
                else:
                    show_frame = processor.combine_overlay(result)
                cv2.imshow(WINDOW_TITLE, show_frame)

                if writer is not None and not paused:
                    writer.write(show_frame)

                if processor.trigger_pause:
                    print("@@ Detector triggered pause. Space to continue.")
                    processor.trigger_pause = False
                    paused = True

                end_loop_s = time.perf_counter()
                loop_s = end_loop_s - start_loop_s

                key = cv2.waitKey(fps_ms) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = not paused
                elif key == ord('o'):
                    view_org = not view_org
                elif key == ord('s'):
                    self.skip_num = 1 + self.skip_num % 4
                elif key == ord('1'):
                    self.zoom = 1
                    cv2.resizeWindow(WINDOW_TITLE, width, height)
                elif key == ord('2'):
                    self.zoom = 2
                    cv2.resizeWindow(WINDOW_TITLE, width//2, height//2)
                elif key == ord('3'):
                    self.zoom = 3
                    cv2.resizeWindow(WINDOW_TITLE, width//3, height//3)
                elif key == ord('4'):
                    self.zoom = 4
                    cv2.resizeWindow(WINDOW_TITLE, width//4, height//4)
        finally:
            if writer is not None:
                writer.release()
                if processor is not None:
                    processor.exportJson(detector_path)
            cap.release()
            cv2.destroyAllWindows()

        print("@@ end")

if __name__ == "__main__":
    m = Main()
    m.run()

