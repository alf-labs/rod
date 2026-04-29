import cv2
import math
import numpy as np
import re
import scipy
from processor import ProcessorBase

class TopTracker(ProcessorBase):
    def __init__(self, start_frame):
        super().__init__()
        self.start_frame = start_frame
        self.tracker_template = None

    def init_size(self, width, height):
        super().init_size(width, height)

    def init_overlay(self, frame):
        super().init_overlay(frame)

    def filter(self, window_title, frame_index, frame):
        if self.tracker_template == None:
            while not self.select_roi(window_title, frame_index, frame):
                print(f"@@ [frame #{frame_index}] Need a valid top coupler area to continue.")
        return super().filter(window_title, frame_index, frame)

    def select_roi(self, window_title, frame_index, frame):
        super().select_roi_called()
        print(f"@@ [frame #{frame_index}] Select top coupler area. 'c' to cancel, space/return to accept.")
        rect = cv2.selectROI(window_title, frame, showCrosshair=True, fromCenter=False)
        # Result rect should be empty if canceled.
        print(f"@@ [frame #{frame_index}] Result: ", repr(rect))
        # CV2 uses a tuple instead of a cv::Rect object so there's no .empty() method
        x, y, w, h = rect
        is_empty = w <= 0 or h <= 0
        return not is_empty

    def export(self):
        return super().export()

    def release(self):
        super().release()

