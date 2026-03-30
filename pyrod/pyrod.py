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
    import skimage
    from flask import Flask, render_template, Response, request, jsonify
    from process_locator import LocatorGen, LocatorRdr
    from process_detector import Detector
except ModuleNotFoundError as e:
    print(f"ERROR: Missing library. {e}")
    print( "To fix: $ pip install opencv-python numpy scipy scikit-image imutils flask")
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
        self.debug = True
        self.zoom = 1
        self.skip_num = 1
        self.paused = False
        self.single_frame = False
        self.view_org = False
        self.allow_export = False
        self.quit_requested = False
        self.start_frame = 0
        self.end_loop_frame = 0
        self.processors = []
        self.export_content = {}
        self.locator_path = ""
        self.export_path = ""
        self.do_crop = False

    def print_fps(self, loop_s, frame_count, dest):
        fps = 1/loop_s if loop_s > 0 else 0
        ms = int(loop_s * 1000)
        text = f"[{frame_count:05d}] {self.mx:03d} x, {ms} ms, {fps:.2f} fps"
        if self.debug:
            z = self.zoom
            cv2.putText(dest, text,
                (10 * z, 30 * z),           # bottom-left coord
                cv2.FONT_HERSHEY_DUPLEX,    # font
                z,                          # font scale
                (0, 255, 255),              # color
                z )                         # line thickness
        elif frame_count % 10 == 0:
            print(text)

    def parseArgs(self):
        parser = argparse.ArgumentParser(description="PyRod")
        parser.add_argument("-d", "--no-debug", action="store_true", help="Disable Debug Overlay")
        parser.add_argument("-i", "--input", default="0", help="Input video")
        parser.add_argument("-o", "--output", default=OUT_VIDEO_FILE_PATH, help="Output video")
        parser.add_argument("-n", "--no-video", action="store_true", help="Skip Video Output")
        parser.add_argument("-l", "--locator", default="", help="Locator JSON data to read back")
        parser.add_argument("-0", "--locator-only", action="store_true", help="Only run locator process")
        parser.add_argument("-1", "--detector-preview", action="store_true", help="Run detector in preview (no inpaint)")
        parser.add_argument("-c", "--crop", action="store_true", help="Center Crop Large Video to 1920x1080")
        parser.add_argument("-s", "--start", default="0", help="Start frame")
        parser.add_argument("-e", "--end", default="0", help="End/loop frame")
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
        self.export_path = self.output_path.replace(".mp4", "") + ".json"
        print("Input:", self.input_path)
        print("Output:", self.output_path, "(disabled by -n)" if args.no_video else "")
        if args.start.isdigit():
            self.start_frame = int(args.start)
        if args.end.isdigit():
            self.end_loop_frame = int(args.end)
        self.do_crop = args.crop
        return args

    def parseKeys(self, processor, wait_ms=1):
        if self.paused:
            wait_ms = 300
        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord('q'):
            self.quit_requested = True
        elif key == ord(' '):
            self.single_frame = False
            self.paused = not self.paused
        elif key == ord('e'):
            self.single_frame = True
            self.paused = False
        elif key == ord('o'):
            self.view_org = not self.view_org
        elif key == ord('m'):
            if processor is not None:
                processor.view_mask = not processor.view_mask
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
            processor.pre_release()
            if self.allow_export:
                self.export_content.update( processor.export() )
            processor.release()
        processor = None
        processor_idx += 1
        if processor_idx < len(self.processors):
            print(f"@@ Switch to next processor #{processor_idx}")
            processor = self.processors[processor_idx]
        else:
            print(f"@@ No next processor #{processor_idx}")
        return processor, processor_idx

    def read_json(self, filename):
        print(f"@@ Load JSON from {filename}")
        with open(filename, "r") as f:
            self.export_content = json.load(f)

    def export_json(self, filename, content):
        with open(filename, "w") as f:
            json.dump(content, f, indent=2)
        print(f"@@ Export JSON output to {filename}")

    def run(self):
        print("@@ Run")
        args = self.parseArgs()

        if args.locator:
            self.read_json(self.locator_path)
            if "pyrod" in self.export_content:
                pyrod_data = self.export_content["pyrod"]
                self.input_path = pyrod_data["input_path"]
                self.start_frame = pyrod_data["start_frame"]
                self.end_loop_frame = pyrod_data["end_frame"]
                self.do_crop = not not pyrod_data["crop"]
                print(f"Overriding input to file '{self.input_path}', frames {self.start_frame} to {self.end_loop_frame}, {'cropped' if self.do_crop else 'uncropped'}")

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
            vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"@@ Opening input {vid_width}x{vid_height}@{fps} fps")
            fps_ms = int(1000 / fps)
            loop_s = 0
            init_once = True
            processor = None
            self.debug = not args.no_debug
            self.paused = False

            do_crop = self.do_crop
            cropped_width = vid_width
            cropped_height = vid_height
            if do_crop:
                cropped_width = min(1920, vid_width)
                cropped_height = min(1080, vid_height)
                crop_x1 = (vid_width - cropped_width) // 2
                crop_y1 = (vid_height - cropped_height) // 2
                crop_x2 = crop_x1 + cropped_width
                crop_y2 = crop_y1 + cropped_height
            width = cropped_width
            height = cropped_height

            # Processor #0
            processor_idx = 0
            if args.locator:
                loc_reader = LocatorRdr()
                self.processors.append( loc_reader )
                loc_reader.read_json(self.export_content["locator"])
            else:
                self.processors.append( LocatorGen() )
            processor = self.processors[0]
            if not args.locator_only:
                # Processor #1
                if args.detector_preview:
                    self.processors.append( Detector(processor, inpainting=False) )
                else:
                    self.processors.append( Detector(processor) )
            for p in self.processors:
                p.debug = self.debug
            print(f"@@ Start with processor #{processor_idx}: {processor}")

            if args.no_video == False:
                self.allow_export = True
                writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height), isColor=True)
                self.export_content["pyrod"] = {
                    "input_path":   self.input_path,
                    "start_frame":  self.start_frame,
                    "end_frame":    self.end_loop_frame,
                    "crop":         self.do_crop
                }
                print(f"@@ Writing {width}x{height}@{fps} fps to", self.output_path)

            last_frame = None
            end_reached = False
            frame_count = self.start_frame
            do_downscale = 1  # can be either 1 or 2
            print(f"@@ Reader cap isOpened: {cap.isOpened()}, {width}x{height}@{fps} fps")
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            while cap.isOpened() and not self.quit_requested:
                self.parseKeys(processor)
                start_loop_s = time.perf_counter()
                if self.paused:
                    frame = last_frame.copy()
                else:
                    if init_once:
                        print(f"Init Processor {processor_idx}, Video size: {width}x{height}")
                        processor.init_size(width, height)
                        do_downscale = processor.downscale
                        width = processor.width
                        height = processor.height
                        init_once = False

                    if end_reached:
                        print(f"@@ capture loop next")
                        end_reached = False
                        processor, processor_idx = self.next_processor(processor, processor_idx)
                        if processor is not None:
                            init_once = True
                            frame_count = self.start_frame
                            width = cropped_width
                            height = cropped_height
                            self.paused = False
                            continue
                        else:
                            print(f"@@ capture loop ended")
                            break
                    ret, frame = cap.read()
                    if not ret or processor.next_processor_requested:
                        # If video recorded file ends, loop back to the beginning
                        print(f"@@ capture end reached?")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
                        end_reached = True
                        continue
                    if do_crop:
                        frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    if do_downscale == 2:
                        frame = cv2.pyrDown(frame) # downscale by a fixed 2x factor
                    frame_count += 1
                    if self.end_loop_frame > 0 and self.end_loop_frame == frame_count:
                        end_reached = True
                    last_frame = frame.copy()

                _skip_num = self.skip_num
                if _skip_num > 1:
                    if frame_count % _skip_num != 0:
                        continue

                if not self.paused:
                    processor.init_overlay(frame)
                    result = processor.filter(frame_count, frame)

                    self.print_fps(loop_s, frame_count, processor.overlay)

                if self.debug:
                    if self.view_org:
                        show_frame = processor.combine_overlay(frame)
                    else:
                        show_frame = processor.combine_overlay(result)
                else:
                    show_frame = result
                cv2.imshow(WINDOW_TITLE, show_frame)

                if self.single_frame:
                    self.paused = True

                if self.allow_export and not self.paused:
                    if do_downscale == 2:
                        show_frame = cv2.resize(show_frame, (0,0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
                    writer.write(show_frame)

                if processor.trigger_pause:
                    print("@@ Detector triggered pause. Space to continue.")
                    processor.trigger_pause = False
                    self.paused = True

                end_loop_s = time.perf_counter()
                loop_s = end_loop_s - start_loop_s

        finally:
            print("@@ Main loop ended.")
            if writer is not None:
                writer.release()
            if self.allow_export:
                self.export_json(self.export_path, self.export_content)
            self.next_processor(processor, processor_idx)
            cap.release()
            cv2.destroyAllWindows()

        print("@@ end")

if __name__ == "__main__":
    m = Main()
    m.run()

