import cv2
import math
import numpy as np
from processor import ProcessorBase
from point import Point
from rect import Rect
from process_coupler import ROI_WIDTH_PCT

SEARCH_WIDTH_PCT = 3

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
        if self.current_template is None:
            self.current_template = self.tracker_templates[coupler.coupler_ref]
        coupler_template = self.current_template
        cr = coupler_template.rect.copy()
        cr.recenter_to(coupler.center.x, coupler.center.y)
        srect = self.get_search_window(w, h, coupler_template)

        sr = cr.copy()
        sr.move_by(0, cr.h)
        sr.scale_by(SEARCH_WIDTH_PCT, 1.0)

        if self.compute_overlay:
            # print(f"@@ [{frame_index:04d} {cr} center {cr.center()} // {sr} // {srect}]")
            self.draw_rect(srect, (255, 255, 0))
            self.draw_rect(cr,    (255, 128, 0))

        while sr.y+sr.h < h:
            sr_lu = lu[sr.y : sr.y + sr.h, sr.x : sr.x + sr.w]
            cv_lu = self.get_cv_vectorized(sr_lu)
            # cv_lu = np.convolve(cv_lu, self.cv_smooth_kernel, mode="same")
            cv_lu_inv = 1 - cv_lu
            cv_lu_inv = cv_lu_inv ** 4
            if self.compute_overlay:
                self.draw_rect(sr,    (  0, 255, 0), width=1)
                self.draw_curve(cv_lu_inv * sr.h, sr, (0, 255, 255))
            sr.move_by(0, sr.h)

        return frame

    def draw_rect(self, rect, color, width=2):
        x = rect.x
        y = rect.y
        w = rect.w
        h = rect.h
        cv2.rectangle(self.overlay, (x, y), (x + w - 1, y + h - 1), color, width)

    def draw_curve(self, data, rect, color, width=1):
        n = len(data)
        fx = rect.w / n
        x1 = rect.x
        y1 = rect.y + rect.h
        ox = x1
        oy = int(y1 - data[0])
        for k in range(1, n):
            nx = int(x1 + k * fx)
            ny = int(y1 - data[k])
            cv2.line(self.overlay, (ox, oy), (nx, ny), color, width)
            ox = nx
            oy = ny

    def get_search_window(self, width, height, coupler_template):
        template_rect = coupler_template.rect.copy()
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

    def get_cv_vectorized(self, strip):
        """
        Coefficient of Variation (CV)
        Calculates CV for all columns in a strip simultaneously.
        'strip' should be a (self.num_bottom_rows_cv, width) array.
        """
        # Convert to float32 for math
        data = strip.astype(np.float32)

        # Calculate mean and std across the vertical axis (axis 0)
        means = np.mean(data, axis=0)
        stds = np.std(data, axis=0)

        # Avoid division by zero: where mean is 0, CV is 0
        # Using np.divide with 'where' condition handles this cleanly
        cv_array = np.divide(stds, means, out=np.zeros_like(stds), where=means > 0)

        return cv_array
