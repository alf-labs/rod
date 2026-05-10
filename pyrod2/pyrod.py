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
import os
import re
import sys
import time

try:
    import cv2
    import numpy as np
    import imutils
    import scipy
    from flask import Flask, render_template, Response, request, jsonify
    from process_coupler import CouplerTracker
    from process_detector import RodDetector
    from process_inpainter import ProcessInpainter, ROD_DILATE_PX, ROD_BLUR_PX
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

OUT_VIDEO_FILE_PATH = "output/NAME_IDX_TIME.mp4"

DISPLAY_NONE = 0
DISPLAY_NO_OVERLAY = 1
DISPLAY_WITH_OVERLAY = 2

class Main:
    def __init__(self):
        self.mx = 0
        self.my = 0
        self.zoom = 1
        self.skip_num = 1
        self.paused = False
        self.single_frame = False
        self.view_org = False
        self.write_json = True
        self.write_video = True
        self.overlay_in_video = False
        self.compute_overlay = True
        self.display_mode = DISPLAY_WITH_OVERLAY
        self.quit_requested = False
        self.start_frame = 0
        self.end_frame = 0
        self.processors = []
        self.export_content = {}
        self.should_export = False
        self.coupler_path = ""
        self.export_path = ""
        self.crop_roi = {}

    def print_fps(self, loop_s, frame_count, dest):
        fps = 1/loop_s if loop_s > 0 else 0
        ms = int(loop_s * 1000)
        text = f"[{frame_count:05d}] {self.mx:03d} x, {ms} ms, {fps:.2f} fps"
        if self.compute_overlay:
            z = self.zoom
            cv2.putText(dest, text,
                (10 * z, 30 * z),           # bottom-left coord
                cv2.FONT_HERSHEY_DUPLEX,    # font
                z,                          # font scale
                (0, 255, 255),              # color
                z )                         # line thickness
        if self.display_mode != DISPLAY_WITH_OVERLAY and frame_count % 100 == 0:
            print(text)

    def parse_args(self):
        parser = argparse.ArgumentParser(description="PyRod")
        parser.add_argument("-d", "--display", default="full", choices=["none", "prod", "full"], help="Window Display")
        parser.add_argument("-i", "--input", default="", help="Input video")
        parser.add_argument("-o", "--output", default=OUT_VIDEO_FILE_PATH, help="Output video")
        parser.add_argument("-n", "--no-video", action="store_true", help="Skip Video Output")
        parser.add_argument(      "--overlay-video", action="store_true", help="Include Overlay in Video Output")
        parser.add_argument("-r", "--roi", default="1280x720+180", help="Center ROI w/ vertical offset")
        parser.add_argument("-s", "--start", default="0", help="Start frame")
        parser.add_argument("-e", "--end", default="0", help="End/loop frame")
        parser.add_argument(      "--no-json", action="store_true", help="Skip JSON Export")

        parser.add_argument("-0", "--coupler-only", action="store_true", help="Only run top-coupler location process")
        parser.add_argument(      "--load-json", default="", help="JSON data to read back")

        parser.add_argument(      "--rod-widths", default="15,40,/1280", help="Detector Rod size top vs bottom")
        parser.add_argument("-p", "--inpaint", default="left", choices=["left", "right", "mix", "none"], help="Inpaint algorithm")
        parser.add_argument(      "--rod-dilate-px", type=int, default=ROD_DILATE_PX, help="Dilate filter kernel after rod detection")
        parser.add_argument(      "--rod-blur-px", type=int, default=ROD_BLUR_PX, help="Blur filter kernel after rod detection")
        args = parser.parse_args()
        self.args = args

        input_idx = ""
        path_name = "output"

        self.input_path = args.input
        if self.input_path.isdigit():
            input_idx = int(self.input_path)
            self.input_path = IN_VIDEOS[input_idx % len(IN_VIDEOS)]

        if args.load_json:
            self.coupler_path = args.load_json
            path_name = re.sub(r"(\D+).*", r"\1", os.path.basename(args.load_json)) # stop at first digit

        self.output_path = f"{args.output}".replace("NAME", path_name)
        self.output_path = self.output_path.replace("IDX", str(input_idx))
        self.output_path = self.output_path.replace("TIME", time.strftime("%Y-%m-%d_%H-%M-%S"))
        self.output_path = re.sub(r"__+", r"_", self.output_path)
        self.export_path = self.output_path.replace(".mp4", "").replace(".MP4", "") + ".json"
        print("Input:", self.input_path)
        print("Output:", self.output_path, "(disabled by -n)" if args.no_video else "")

        self.crop_roi = self.parse_roi(args.roi)
        self.start_frame = self.parse_frame_timestamp(args.start)
        self.end_frame = self.parse_frame_timestamp(args.end)

        self.write_json = not args.no_json
        self.write_video = not args.no_video
        self.overlay_in_video = args.overlay_video
        self.display_mode = {
            "none": DISPLAY_NONE,
            "prod": DISPLAY_NO_OVERLAY,
            "full": DISPLAY_WITH_OVERLAY,
        }.get(args.display)

        return args

    def parse_roi(self, roi_str):
        """ROI str is WIDTHxHEIGHT+YOFFSET"""
        pattern = r"(?P<w>\d+)x(?P<h>\d+)\+(?P<y>\d+)"
        match = re.search(pattern, roi_str)
        assert match is not None, "Expected ROI syntax: WIDTHxHEIGHT+YOFFSET"
        return {
            "width":   int(match.group("w")),
            "height":  int(match.group("h")),
            "yoffset": int(match.group("y")),
        }

    def parse_frame_timestamp(self, ts_str):
        if ts_str.isdigit():
            # Just a number: this represents a frame number
            return int(ts_str)
        # The following formats represent an hour:minute:second timestamp.
        #   "12:34:45" or "34:45"
        #   "1h23m45s" or "23m45s" or "45s"
        # We just return a tuple (NN, "seconds") and will convert it to an integer once
        # the know the frame rate of the input video.
        pattern1 = r"^(?:(?P<h>\d+):)?(?P<m>\d{1,2}):(?P<s>\d{1,2})$"
        match = re.search(pattern1, ts_str)
        if match is None:
            pattern2 = r"^(?:(?P<h>\d+)h)?(?:(?P<m>\d{1,2})m)(?P<s>\d{1,2})s$"
            match = re.search(pattern1, ts_str)
        if match:
            h = match.group("h") or 0
            m = match.group("m") or 0
            s = match.group("s") or 0
            return ( int(h) * 3600 + int(m) * 60 + int(s), "seconds" )

    def parse_keys(self, processor, wait_ms=1):
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
            cv2.resizeWindow(WINDOW_TITLE, self.width, self.height)
        elif key == ord('2'):
            self.zoom = 2
            cv2.resizeWindow(WINDOW_TITLE, self.width//2, self.height//2)
        elif key == ord('3'):
            self.zoom = 3
            cv2.resizeWindow(WINDOW_TITLE, self.width//3, self.height//3)
        elif key == ord('4'):
            self.zoom = 4
            cv2.resizeWindow(WINDOW_TITLE, self.width//4, self.height//4)

    def next_processor(self, processor, processor_idx):
        if processor is not None:
            processor.pre_release()
            if self.write_json:
                new_export = processor.export()
                self.should_export = self.should_export or (len(new_export) > 0)
                self.export_content.update( new_export )
            processor.release()
        processor = None
        processor_idx += 1
        if processor_idx < len(self.processors):
            print(f"@@ Switch to next processor #{processor_idx}")
            processor = self.processors[processor_idx]
        else:
            print(f"@@ No next processor #{processor_idx}")
        return processor, processor_idx

    def read_json_file(self, filename):
        print(f"@@ Load JSON from {filename}")
        with open(filename, "r") as f:
            self.export_content = json.load(f)

    def write_json_file(self, filename, content):
        with open(filename, "w") as f:
            json.dump(content, f, indent=2)
        print(f"@@ Export JSON output to {filename}")

    def run(self):
        print("@@ Run")
        args = self.parse_args()

        stats_start_main_s = time.perf_counter()
        stats_iterations = 0

        if self.coupler_path:
            self.read_json_file(self.coupler_path)
            if "pyrod" in self.export_content:
                pyrod_data = self.export_content["pyrod"]
                self.input_path = pyrod_data["input_path"]
                if "start_frame" in pyrod_data:
                    start = pyrod_data["start_frame"]
                    self.start_frame = int(start)
                if "end_frame" in pyrod_data:
                    end = pyrod_data["end_frame"]
                    self.end_frame = int(end)
                self.crop_roi = pyrod_data["crop_roi"]
                print(f"Overriding input to file '{self.input_path}', frames {self.start_frame} to {self.end_frame}, {'cropped' if self.crop_roi else 'uncropped'}")

        display_mode = self.display_mode
        def _mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_MOUSEMOVE:
                self.mx = x
                self.my = y
        mouse_callback = None
        if display_mode != DISPLAY_NONE:
            cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_TITLE, 1280//2, 720//2)
            mouse_callback = _mouse_callback
            cv2.setMouseCallback(WINDOW_TITLE, mouse_callback)

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
            self.compute_overlay = display_mode == DISPLAY_WITH_OVERLAY or self.overlay_in_video
            self.paused = False

            if isinstance(self.start_frame, tuple):
                print(f"@@ Start frame: {self.start_frame} --> frame {self.start_frame[0] * fps}, fps {fps}")
                self.start_frame = int(self.start_frame[0] * fps)
            else:
                print(f"@@ Start frame: {self.start_frame}, fps {fps}")
            if isinstance(self.end_frame, tuple):
                print(f"@@ End frame: {self.end_frame} --> frame {self.end_frame[0] * fps}, fps {fps}")
                self.end_frame = int(self.end_frame[0] * fps)
            else:
                print(f"@@ End frame: {self.end_frame}, fps {fps}")

            do_crop = False
            cropped_width = vid_width
            cropped_height = vid_height
            if vid_width > self.crop_roi["width"]:
                do_crop = True
                cropped_width = min(self.crop_roi["width"], vid_width)
                cropped_height = min(self.crop_roi["height"], vid_height)
                y_offset = self.crop_roi["yoffset"]
                crop_x1 = (vid_width - cropped_width) // 2
                crop_y1 = (vid_height - cropped_height) // 2 + y_offset
                crop_x2 = crop_x1 + cropped_width
                crop_y2 = crop_y1 + cropped_height
                print(f"@@ Crop bounds: {crop_x1}x{crop_y1} -> {crop_x2}x{crop_y2}")
            width = cropped_width
            height = cropped_height
            self.width = width
            self.height = height

            # Processor #0
            processor_idx = 0
            tracker = CouplerTracker(
                    start_frame=self.start_frame,
            )
            self.processors.append( tracker )
            if self.coupler_path:
                tracker.read_json(self.export_content)
            processor = self.processors[0]

            if not args.coupler_only:
                # Processor #1
                detector = RodDetector(
                    tracker,
                    rod_widths_str=args.rod_widths
                )
                self.processors.append( detector )
                if self.coupler_path:
                    detector.read_json(self.export_content)

                # Processor #2
                inpainter = ProcessInpainter(
                    tracker,
                    detector,
                    inpainting=args.inpaint,
                    rod_dilate_px=args.rod_dilate_px,
                    rod_blur_px=args.rod_blur_px,
                )
                self.processors.append( inpainter )

            for p in self.processors:
                p.compute_overlay = self.compute_overlay
            print(f"@@ Start with processor #{processor_idx}: {processor}")

            if self.write_video:
                writer = cv2.VideoWriter(self.output_path, fourcc, fps, (vid_width, vid_height), isColor=True)
                print(f"@@ Writing {vid_width}x{vid_height}@{fps} fps to", self.output_path)
            if self.write_json:
                self.export_content["pyrod"] = {
                    "input_path":   self.input_path,
                    "start_frame":  self.start_frame,
                    "end_frame":    self.end_frame,
                    "crop_roi":     self.crop_roi,
                }

            last_frame = None
            end_reached = False
            frame_count = self.start_frame
            do_downscale = 1  # can be either 1 or 2
            print(f"@@ Reader cap isOpened: {cap.isOpened()}, {width}x{height}@{fps} fps")
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            while cap.isOpened() and not self.quit_requested:
                stats_iterations += 1
                self.parse_keys(processor)
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
                        uncropped_frame = frame
                        frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    if do_downscale == 2:
                        frame = cv2.pyrDown(frame) # downscale by a fixed 2x factor
                    last_frame = frame.copy()

                _skip_num = self.skip_num
                if _skip_num > 1:
                    if frame_count % _skip_num != 0:
                        continue

                if not self.paused:
                    processor.init_overlay(frame)
                    if do_crop:
                        cv2.rectangle(processor.overlay, (0, 0), (cropped_width - 1, cropped_height - 1), (0,0,0), 1)

                    result = processor.filter(WINDOW_TITLE, frame_count, frame)

                    if processor.select_roi_invoked and mouse_callback:
                        # cv2.selectROI changes the mouse callback so we need to restore it
                        cv2.setMouseCallback(WINDOW_TITLE, mouse_callback)
                        processor.select_roi_invoked = False

                    self.print_fps(loop_s, frame_count, processor.overlay)

                original_frame = result
                overlaid_frame = None

                if display_mode == DISPLAY_NONE:
                    pass
                elif display_mode == DISPLAY_NO_OVERLAY:
                    cv2.imshow(WINDOW_TITLE, original_frame)
                elif display_mode == DISPLAY_WITH_OVERLAY:
                    if self.view_org:
                        overlaid_frame = processor.combine_overlay(frame)
                    else:
                        overlaid_frame = processor.combine_overlay(original_frame)
                    cv2.imshow(WINDOW_TITLE, overlaid_frame)

                if self.write_video and not self.paused:
                    if self.overlay_in_video:
                        video_frame = overlaid_frame
                        if video_frame is None:
                            video_frame = processor.combine_overlay(original_frame)
                    else:
                        video_frame = original_frame
                    if do_downscale == 2:
                        video_frame = cv2.resize(video_frame, (0,0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
                    if do_crop:
                        uncropped_frame[crop_y1:crop_y2, crop_x1:crop_x2] = video_frame
                        video_frame = uncropped_frame
                    writer.write(video_frame)

                if self.single_frame:
                    self.paused = True

                if processor.trigger_pause:
                    print("@@ Detector triggered pause. Space to continue.")
                    processor.trigger_pause = False
                    self.paused = True

                if not self.paused:
                    frame_count += 1
                    if self.end_frame > 0 and self.end_frame == frame_count:
                        end_reached = True

                end_loop_s = time.perf_counter()
                loop_s = end_loop_s - start_loop_s

        finally:
            print("@@ Main loop ended.")
            self.next_processor(processor, processor_idx)
            if writer is not None:
                writer.release()
            if self.write_json and self.should_export:
                self.write_json_file(self.export_path, self.export_content)
            cap.release()
            if display_mode != DISPLAY_NONE:
                cv2.destroyAllWindows()

        stats_end_main_s = time.perf_counter()
        stats_duration_s = int(stats_end_main_s - stats_start_main_s)
        stats_iterations = max(1, stats_iterations)
        stats_ms = 1000.0 * stats_duration_s / stats_iterations
        stats_fps = stats_iterations / stats_duration_s if stats_duration_s > 0 else 0
        print(f"@@ Stats: {stats_iterations} frames in {stats_duration_s // 60} min {stats_duration_s % 60} sec; {stats_ms:.2f} ms/frame; {stats_fps:.2f} fps")

        print("@@ end")

if __name__ == "__main__":
    m = Main()
    m.run()

