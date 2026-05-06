import cv2
import math
import numpy as np
from processor import ProcessorBase
from point import Point
from rect import Rect
from process_coupler import ROI_WIDTH_PCT

SEARCH_WIDTH_PCT = 3
ROD_W_TOP = 15 / 1280
ROD_W_BOT = 40 / 1280

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
        self.rod_w_top = int(ROD_W_TOP * width)
        self.rod_w_bot = int(ROD_W_BOT * width)

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
            self.draw_rect(cr,    (255, 128, 0))

        # Experiment 1: Use a few lines to run a CV computation, and display it.
        # self.experimental_cv_search(w, h, lu, cr)

        # Experiment 2: Re-implement the old Lua process
        self.experimental_lua_search(frame_index, w, h, lu, cr)

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
        y1 = rect.y + rect.h - int(y)
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

    def experimental_lua_search(self, frame_index, w, h, lu, cr):
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
        # we'll search for the rod close to the center at first.
        # TBD: reuse data from the last frame as it should be "close by".
        rod_center_x = (x1 + x2) // 2
        rod_w = self.rod_w_top

        for y1 in range(yt, yb):
            # The ideal rod width varies per line
            ideal_rod_w = self.rod_w_top + (self.rod_w_bot - self.rod_w_top) / (yb - yt) * (y1 - yt)

            # For testing we just look at a single line (it's a 1,N 2d array though)
            # Note: do not run cv2.normalize(0..255). It's idempotent with the percentile threshold below.
            y_lu = lu[y1 : y1 + 1, x1 : x2]
            # ravel() flattens 2d --> 1d
            flat = y_lu.ravel()

            # Select everything that is higher than the 80th percentile.
            f_threshold = np.percentile(flat, 80)
            selected = (flat >= f_threshold).astype(np.uint8)
            run_cw = self.select_best_run(selected, self.rod_w_top, self.rod_w_bot, rod_center_x - x1, ideal_rod_w)

            if run_cw is not None:
                rod_center_x = x1 + run_cw[0]
                rod_w = run_cw[1]
                run_color = (0, 165, 255)

            # debug
            if self.compute_overlay:
                # display current run
                if run_cw is not None:
                    lx1 = int( rod_center_x - rod_w / 2)
                    lx2 = int( rod_center_x + rod_w / 2)
                    cv2.line(self.overlay, (lx1, y1), (lx2, y1), run_color, 1)
                # display the curve at the bottom of the SR rect
                if y1 == sr.y + sr.h - 1:
                    self.draw_rect(sr,    (  0, 255, 0), width=1)
                    self.draw_curve(flat / 255 * sr.h, sr, (0, 255, 255))
                    self.draw_curve(selected * sr.h, sr, (255, 0, 0))
                    self.draw_threshold(f_threshold / 255 * sr.h, sr, (0, 165, 255)) # orange
                    sr.move_by(0, sr.h)

    def select_best_run(self, selected, min_size, max_size, rod_center_x, ideal_rod_w):
        # Find transitions (0 to 1 and 1 to 0)
        # Prepend/Append 0 to handle runs at the very start or end
        padded = np.pad(selected, (1, 1), 'constant', constant_values=0)
        diffs = np.diff(padded)

        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == 255)[0]    # -1 on an uint8 input array
        lengths = ends - starts

        # Identify which runs to keep
        valid_indices = np.where((lengths >= min_size) & (lengths <= max_size))[0]

        target_x1 = rod_center_x - ideal_rod_w / 2
        target_x2 = rod_center_x + ideal_rod_w / 2
        best_score = 0
        selected_cw = None  # center + width

        # Select the one that is the closest to the desired rod center
        for idx in valid_indices:
            x1 = starts[idx]
            x2 = ends[idx]

            # the score is the IoU of this segment vs the target
            score = self.iou(x1, x2, target_x1, target_x2)
            if score > best_score:
                best_score = score
                selected_cw = ( (x1 + x2) / 2, (x2 - x1) )

        return selected_cw

    def iou(self, ax1, ax2, bx1, bx2):
        """Computes IoU (Intersection over Union) between 2 segments A and B"""
        intersection = max(0, min(ax2, bx2) - max(ax1, bx1))
        union = (ax2 - ax1) + (bx2 - bx1) - intersection
        return intersection / union if union > 0 else 0


# ~~
