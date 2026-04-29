import cv2
import math
import numpy as np
import re
import scipy
from processor import ProcessorBase
from rect import Rect
from tracker_template import TrackerTemplate


class TopTracker(ProcessorBase):
    def __init__(self, start_frame):
        super().__init__()
        self.start_frame = start_frame
        self.current_template = None
        self.tracker_templates = {}

    def init_size(self, width, height):
        super().init_size(width, height)

    def init_overlay(self, frame):
        super().init_overlay(frame)

    def filter(self, window_title, frame_index, frame):
        h, w = frame.shape[:2]
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lu = lab[:, :, 0]
        frame = cv2.cvtColor(lu, cv2.COLOR_GRAY2BGR)

        while self.current_template == None:
            print(f"@@ [frame #{frame_index}] Select a valid top coupler area to continue.")
            self.current_template = self.select_roi(window_title, frame_index, frame, lu)
            if self.current_template:
                self.tracker_templates[frame_index] = self.current_template.copy()

        search_rect = self.get_search_window(w, h)
        search_lu = lu[search_rect.y : search_rect.y + search_rect.h, search_rect.x : search_rect.x + search_rect.w]

        print(f"@@ DEBUG track {self.current_template.rect} in search {search_rect}")
        print(f"@@ DEBUG track {self.current_template.template.shape} in search {search_lu.shape}")

        res = cv2.matchTemplate(search_lu, self.current_template.template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # Update current template with best match
        self.current_template.rect.x = max_loc[0] + search_rect.x
        self.current_template.rect.y = max_loc[1] + search_rect.y

        self.draw_rect(search_rect, (0, 0, 255))
        self.draw_rect(self.current_template.rect, (0, 255, 0))
        print(f"@@ [frame #{frame_index}] --> track {self.current_template.rect}")

        return frame

    def select_roi(self, window_title, frame_index, frame, lu):
        """Returns None if no ROI selected, otherwise returns a Template tuple."""
        super().select_roi_called()
        # print(f"@@ [frame #{frame_index}] Select top coupler area. 'c' to cancel, space/return to accept.")
        rect = cv2.selectROI(window_title, frame, showCrosshair=True, fromCenter=False)
        # Result rect should be empty if canceled.
        print(f"@@ [frame #{frame_index}] Result: ", repr(rect))
        # CV2 uses a tuple instead of a cv::Rect object so there's no .empty() method
        x, y, w, h = rect
        rect = Rect(x, y, w, h)
        if rect.is_empty():
            return None
        else:
            return TrackerTemplate(
                frame_index,
                rect,
                lu[y : y + h, x : x + w].copy()
            )

    def draw_rect(self, rect, color, width=2):
        x = rect.x
        y = rect.y
        w = rect.w
        h = rect.h
        cv2.rectangle(self.overlay, (x, y), (x + w - 1, y + h - 1), color, width)

    def get_search_window(self, width, height):
        template_rect = self.current_template.rect
        c = template_rect.center()
        w = template_rect.w
        h = template_rect.h
        x = c[0] - w  # * 2 / 2
        y = c[1] - h  # * 2 / 2
        w *= 2
        h *= 2
        if x < 0:
            x = 0
        elif x + w >= width:
            x = width - w
        if y < 0:
            y = 0
        elif y + h >= height:
            y = height - h
        return Rect(x, y, w, h)


    def export(self):
        return super().export()

    def release(self):
        super().release()

