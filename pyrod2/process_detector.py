import cv2
import math
import numpy as np
from processor import ProcessorBase
from point import Point
from rect import Rect

ROI_WIDTH_PCT = 1/3
QUALITY_THRESHOLD = 0.1

class RodDetector(ProcessorBase):
    def __init__(self, coupler_tracker):
        super().__init__()
        self.coupler_tracker = coupler_tracker
        self.start_frame = coupler_tracker.start_frame
        self.tracker_templates = coupler_tracker.tracker_templates
        self.couplers = coupler_tracker.couplers
        self.current_template = None
        self.current_search_rect = None

    def init_size(self, width, height):
        super().init_size(width, height)

    def init_overlay(self, frame):
        super().init_overlay(frame)

    def filter(self, window_title, frame_index, frame):
        h, w = frame.shape[:2]
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lu = lab[:, :, 0]
        frame = cv2.cvtColor(lu, cv2.COLOR_GRAY2BGR)

        coupler = self.couplers[frame_index]
        if coupler is None:
            return frame
        coupler_template = self.tracker_templates[coupler.coupler_ref]

        srect = self.current_search_rect
        if srect is None:
            srect = self.current_search_rect = self.get_search_window(w, h, coupler_template)
        if self.current_template is None:
            r = coupler_template.rect.copy()
            r.move_by(0, r.h)
            self.current_template = lu[r.y : r.y + r.h, r.x : r.x + r.w].copy()

        search_lu = lu[srect.y : srect.y + srect.h, srect.x : srect.x + srect.w]
        res = cv2.matchTemplate(search_lu, self.current_template, cv2.TM_CCOEFF_NORMED)

        if self.compute_overlay:
            color = (0, 255, 255)  # debug search frame is yellow
            self.draw_rect(srect, color)

            res_clipped = np.clip(res, 0, 1.0)
            res_8u = (res_clipped * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(res_8u, cv2.COLORMAP_JET)
            print(f"@@ heatmap shape: {heatmap.shape}")
            rh, rw = heatmap.shape[:2]
            frame[srect.y : srect.y + rh, srect.x : srect.x + rw] = heatmap

        return frame

    def draw_rect(self, rect, color, width=2):
        x = rect.x
        y = rect.y
        w = rect.w
        h = rect.h
        cv2.rectangle(self.overlay, (x, y), (x + w - 1, y + h - 1), color, width)

    def get_search_window(self, width, height, coupler_template):
        template_rect = coupler_template.rect
        c = template_rect.center()
        w = int(ROI_WIDTH_PCT * width)
        h = template_rect.h
        x = c[0] - w // 2
        y = template_rect.y - h
        h = height - y
        if x < 0:
            x = 0
        elif x + w >= width:
            x = width - w
        if y < 0:
            y = 0
        elif y + h >= height:
            y = height - h
        return Rect(x, y, w, h)
