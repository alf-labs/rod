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

        coupler = self.couplers[frame_index]
        if coupler is None:
            return frame
        if self.current_template is None:
            self.current_template = self.tracker_templates[coupler.coupler_ref]
        coupler_template = self.current_template
        # CR: a rect centered on current coupler position, of same w/h as the coupler template.
        cr = coupler_template.rect.copy()
        cr.recenter_to(coupler.center.x, coupler.center.y)

        # SRect: The overall search window (from top of static coupler template to bottom of video)
        # This /could/ be used to only process a sub-area of the original frame for speed purposes
        # (right now it's only used for debug display reference, not for actual processing).
        if self.compute_overlay:
            srect = self.get_search_window(w, h, coupler_template)
            self.draw_rect(srect, (255, 255, 0))

        if self.compute_overlay:
            # print(f"@@ [{frame_index:04d} {cr} center {cr.center()} // {sr} // {srect}]")
            self.draw_rect(cr,    (255, 128, 0))

        # Experiment 1: Use a few lines to run a CV computation, and display it.
        # self.experimental_cv_search(w, h, lu, cr)

        # Experiment 2: Re-implement the old Lua process
        self.experimental_lua_search(w, h, lu, cr)

        # For debug purposes, we display any changes made to the LU image.
        frame = cv2.cvtColor(lu, cv2.COLOR_GRAY2BGR)
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
        oy = y1 - int(data[0])
        for k in range(1, n):
            nx = int(x1 + k * fx)
            ny = y1 - int(data[k])
            cv2.line(self.overlay, (ox, oy), (nx, ny), color, width)
            ox = nx
            oy = ny

    def draw_threshold(self, y, rect, color, width=1):
        x1 = rect.x
        x2 = x1 + rect.w
        y1 = rect.y + rect.h - y
        cv2.line(self.overlay, (x1, y1), (x2, y1), color, width)

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

    def experimental_cv_search(self, w, h, lu, cr):
        """Experiment 1: Use a few lines to run a CV computation, and display it."""

        # SR: The actual search rect. It's located just below the coupler area (cr)
        sr = cr.copy()
        sr.move_by(0, cr.h)
        sr.scale_by(SEARCH_WIDTH_PCT, 1.0)

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

    def experimental_lua_search(self, w, h, lu, cr):
        """Experiment 2: Reimplement the old pixel-based Lua search."""

        # SR: The actual search rect. It's located just below the coupler area (cr)
        sr = cr.copy()
        sr.move_by(0, cr.h)
        sr.scale_by(SEARCH_WIDTH_PCT, 1.0)

        x1 = sr.x
        x2 = sr.x + sr.w
        yt = sr.y
        yb = h
        y_half = (yt + yb) // 2

        for y1 in range(yt, yb):
            # for testing we just look at a single line (it's a 1,N 2d array though)
            y_lu = lu[y1 : y1 + 1, x1 : x2]

            # the Lua algorithm was manually computing the lu min/max and delta.
            # this is basically a normalization.
            if y1 < y_half: # compare the 2 versions
                # Option 1:
                norm_y_lu = cv2.normalize(y_lu, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            else:
                # Option 2: use a percentile to avoid outlier black/white pixels
                p2, p98 = np.percentile(y_lu, [2, 98])
                # Clip values to the percentiles, then stretch
                line_clipped = np.clip(y_lu, p2, p98)
                norm_y_lu = cv2.normalize(line_clipped, None, 0, 255, cv2.NORM_MINMAX)

            # print(f"@@ [{y1}] sr_y_lu = {y_lu}")
            # print(f"@@ [{y1}] norm_y_lu = {norm_y_lu}")
            debug_y_lu = norm_y_lu

            # debug
            if self.compute_overlay:
                # we also place the values back into LU for display
                lu[y1 : y1 + 1, x1 : x2] = debug_y_lu
                # display the curve at the bottom of the SR rect
                if y1 == sr.y + sr.h - 1:
                    self.draw_rect(sr,    (  0, 255, 0), width=1)
                    # ravel() flattens 2d --> 1d
                    flat = norm_y_lu.ravel() / 255 * sr.h
                    self.draw_curve(flat, sr, (0, 255, 255))
                    f_med = int(np.median(flat))
                    self.draw_threshold(f_med, sr, (0, 165, 255)) # orange
                    sr.move_by(0, sr.h)

# ~~
