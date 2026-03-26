#!/usr/bin/python
# Ensure we're running from the Virtual Env version
import os
if not "VIRTUAL_ENV" in os.environ:
    print("ERROR: Run this from venv using 'source ./venv_catd/bin/activate' first")
    exit(1)
IS_RPI = os.path.isfile("/etc/rpi-issue")

import argparse
import base64
import sys
import time

try:
    import cv2
    import numpy as np
    import imutils
    import scipy
    from flask import Flask, render_template, Response, request, jsonify
    from process_locator import LocatorGen, LocatorRdr
    from process_detector import Detector
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

OUT_VIDEO_FILE_PATH = "output/outputIDX_%s.mp4" % time.strftime("%Y-%m-%d_%H-%M-%S")


class Main:
    def __init__(self):
        self.mx = 0
        self.my = 0
        self.zoom = 1
        self.skip_num = 1
        self.pause = False
        self.view_org = False
        self.allow_export = False
        self.quit_requested = False
        self.processors = []
        self.export_path = {}

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

    def parseArgs(self):
        parser = argparse.ArgumentParser(description="PyRod")
        parser.add_argument("-i", "--input", default="0", help="Input video")
        parser.add_argument("-o", "--output", default=OUT_VIDEO_FILE_PATH, help="Output video")
        parser.add_argument("-n", "--no-video", action="store_true", help="Skip Video Output")
        parser.add_argument("-l", "--locator", default="", help="Locator JSON data to read back")
        args = parser.parse_args()
        self.args = args

        input_idx = "_"
        self.input_path = args.input
        if self.input_path.isdigit():
            input_idx = int(self.input_path)
            self.input_path = IN_VIDEOS[input_idx % len(IN_VIDEOS)]
        self.output_path = f"{args.output}".replace("IDX", str(input_idx))
        if args.locator:
            self.locator_path = args.locator
        else:
            self.locator_path = self.output_path.replace(".mp4", "") + ".0.json"
        print("Input:", self.input_path)
        print("Output:", self.output_path, "(disabled by -n)" if args.no_video else "")
        return args

    def parseKeys(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.quit_requested = True
        elif key == ord(' '):
            self.paused = not self.paused
        elif key == ord('o'):
            self.view_org = not self.view_org
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

    def next_processor(self, processor, processor_idx):
        if processor is not None:
            if self.allow_export:
                path = self.export_path.get(processor_idx, None)
                if path:
                    processor.export(path)
            processor.release()
        processor = None
        processor_idx += 1
        if processor_idx < len(self.processors):
            print(f"@@ Switch to next processor #{processor_idx}")
            processor = self.processors[processor_idx]
        else:
            print(f"@@ No next processor #{processor_idx}")
        return processor, processor_idx

    def run(self):
        print("@@ Run")
        args = self.parseArgs()

        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TITLE, 1920//2, 1080//2)
        def _mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_MOUSEMOVE:
                self.mx = x
                self.my = y
        cv2.setMouseCallback(WINDOW_TITLE, _mouse_callback)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        cap = cv2.VideoCapture(self.input_path)
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
            processor = None
            self.paused = False
            self.view_org = True

            # Processor #0
            processor_idx = 0
            if args.locator:
                loc_reader = LocatorRdr()
                self.processors.append( loc_reader )
                loc_reader.readJson(self.locator_path)
            else:
                self.processors.append( LocatorGen() )
                self.export_path[0] = self.locator_path
            processor = self.processors[0]
            # Processor #1
            self.processors.append( Detector(processor) )
            print(f"@@ Start with processor #{processor_idx}: {processor}")

            if args.no_video == False:
                self.allow_export = True
                writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height), isColor=True)
                print(f"@@ Writing {width}x{height}@{fps} fps to", self.output_path)

            last_frame = None
            print(f"@@ Reader cap isOpened: {cap.isOpened()}, {width}x{height}@{fps} fps")
            while cap.isOpened() and not self.quit_requested:
                start_loop_s = time.perf_counter()
                if self.paused:
                    frame = last_frame.copy()
                else:
                    ret, frame = cap.read()
                    if not ret or processor.next_processor_requested:
                        # If video recorded file ends, loop back to the beginning
                        print(f"@@ capture end reached?")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        # Try to read again
                        ret, frame = cap.read()
                        if ret:
                            print(f"@@ capture looped ok")
                            processor, processor_idx = self.next_processor(processor, processor_idx)
                            if processor is not None:
                                init_once = True
                                frame_count = 0
                                self.paused = False
                                self.view_org = True
                        if not ret or processor is None:
                            print(f"@@ capture loop ended")
                            break
                    frame_count += 1
                    last_frame = frame.copy()
                    if self.view_org and frame_count == 50:
                        self.view_org = False

                _skip_num = self.skip_num
                if _skip_num > 1:
                    if frame_count % _skip_num != 0:
                        continue

                if init_once:
                    print(f"Init Processor {processor_idx}, Video size: {width}x{height}")
                    processor.init_size(width, height)
                    init_once = False

                processor.init_overlay(frame)
                result = processor.filter(frame_count, frame)

                self.print_fps(loop_s, processor.overlay)

                if self.view_org:
                    show_frame = processor.combine_overlay(frame)
                else:
                    show_frame = processor.combine_overlay(result)
                cv2.imshow(WINDOW_TITLE, show_frame)

                if self.allow_export and not self.paused:
                    writer.write(show_frame)

                if processor.trigger_pause:
                    print("@@ Detector triggered pause. Space to continue.")
                    processor.trigger_pause = False
                    paused = True

                end_loop_s = time.perf_counter()
                loop_s = end_loop_s - start_loop_s
                self.parseKeys()

        finally:
            print("@@ Main loop ended.")
            if writer is not None:
                writer.release()
            self.next_processor(processor, processor_idx)
            cap.release()
            cv2.destroyAllWindows()

        print("@@ end")

if __name__ == "__main__":
    m = Main()
    m.run()

