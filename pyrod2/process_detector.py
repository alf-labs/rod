import cv2
import math
import numpy as np
import re
from processor import ProcessorBase
from point import Point
from rect import Rect
from process_coupler import ROI_WIDTH_PCT, QUALITY_THRESHOLD
from rod_result import RodResult

SEARCH_WIDTH_PCT = 3
ROD_W_TOP = 15 / 1280
ROD_W_BOT = 40 / 1280

class RodDetector(ProcessorBase):
    def __init__(self, coupler_tracker, rod_widths_str):
        super().__init__()
        self.coupler_tracker = coupler_tracker
        self.start_frame = coupler_tracker.start_frame
        self.tracker_templates = coupler_tracker.tracker_templates
        self.couplers = coupler_tracker.couplers
        self.current_template = None
        self.roi_rect = None
        self.rods = {}
        self.rods_fixed = False
        self.parse_rod_widths_str(rod_widths_str)

    def parse_rod_widths_str(self, rod_widths_str):
        pattern = r"(?P<top>\d+),(?P<bot>\d+),/(?P<width>\d+)"
        match = re.search(pattern, rod_widths_str)
        assert match is not None, "Expected syntax: 'top,bottom,/width', e.g. '15,40,/1280'"
        _top = int(match.group("top"))
        _bot = int(match.group("bot"))
        _width = int(match.group("width"))
        global ROD_W_TOP, ROD_W_BOT
        ROD_W_TOP = _top / _width
        ROD_W_BOT = _bot / _width

    def init_size(self, width, height):
        super().init_size(width, height)
        self.roi_center = width / 2
        self.rod_w_top = int(ROD_W_TOP * width)
        self.rod_w_bot = int(ROD_W_BOT * width)
        print(f"@@ Rod Widths: {self.rod_w_top} to {self.rod_w_bot}")

    def init_overlay(self, frame):
        super().init_overlay(frame)

    def filter(self, window_title, frame_index, frame):
        if ( (frame_index == self.start_frame or self.start_frame == 0)
                and self.rods_fixed
                and len(self.rods) > 0 ):
            self.next_processor_requested = True
            return frame

        h, w = frame.shape[:2]
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lu = lab[:, :, 0]

        coupler = self.couplers[frame_index]
        if coupler is None or coupler.quality < QUALITY_THRESHOLD:
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
            srect = self.get_search_window(cr)
            self.draw_rect(srect, (255, 255, 0))

        if self.compute_overlay:
            self.draw_rect(cr,    (255, 128, 0))

        has_result = frame_index in self.rods

        if not has_result:
            result = self.search_poly_rod(frame_index, w, h, lu, cr)
            self.rods[frame_index] = result
            self.rods_fixed = False
        else:
            # Reuse previous result
            result = self.rods[frame_index]
            # print(f"@@ [{frame_index:04d}] Reuse ROD {result}")

        if self.compute_overlay and result is not None:
            self.draw_rod(result, (0, 165, 255), width=1)

        # # For debug purposes, we display any changes made to the LU image.
        # frame = cv2.cvtColor(lu, cv2.COLOR_GRAY2BGR)
        return frame

    def draw_rod(self, rod_result, color, width=1):
        for y1 in range(rod_result.y_top, rod_result.y_bottom):
            lc = rod_result.poly_c(y1)
            lw = rod_result.poly_w(y1)
            lx1 = int(lc - lw / 2)
            lx2 = int(lc + lw / 2)
            cv2.line(self.overlay, (lx1, y1), (lx2, y1), color, width)

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

    def get_search_window(self, coupler_rect):
        width = self.width
        height = self.height
        c = coupler_rect.center()
        w = int(ROI_WIDTH_PCT * width)
        if self.roi_rect is None:
            y = coupler_rect.y - coupler_rect.h
        else:
            y = self.roi_rect.y
        x = int(self.roi_center - w // 2)
        h = height - y
        self.roi_center = self.roi_center * 0.9 + c[0] * 0.1
        if x < 0:
            x = 0
        elif x + w >= width:
            x = width - w
        if y < 0:
            y = 0
        elif y + h >= height:
            y = height - h
        rect = self.roi_rect = Rect(x, y, w, h)
        return rect

    def search_poly_rod(self, frame_index, w, h, lu, cr):
        """Experiment 2: Reimplement the old pixel-based Lua search but with a numpy take."""

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
        ideal_rod_w = self.rod_w_top
        initial_rod_center = (x1 + x2) // 2
        rod_center = initial_rod_center
        # rod_bounds is (0=y, 1=xcenter, 2=width, 3=xleft, 4=xright). Remove later what we don't need.
        rod_bounds = ( yt, rod_center, ideal_rod_w, int(rod_center - ideal_rod_w / 2), int(rod_center + ideal_rod_w / 2) )

        all_bounds = []
        for y1 in range(yt, yb):
            # The ideal rod width varies per line
            ideal_rod_w = self.rod_w_top + (self.rod_w_bot - self.rod_w_top) / (yb - yt) * (y1 - yt)

            # For testing we just look at a single line (it's a 1,N 2d array though)
            # Note: do not run cv2.normalize(0..255). It's idempotent with the percentile threshold below.
            y_lu_u8 = lu[y1, x1 : x2]

            # Select everything that is higher than the 80th percentile.
            f_threshold = np.percentile(y_lu_u8, 80)
            selected_01 = (y_lu_u8 >= f_threshold).astype(np.uint8)
            run_bounds = self.select_best_run(selected_01, self.rod_w_top, self.rod_w_bot, rod_center - x1, ideal_rod_w)

            if run_bounds is not None:
                run_x1 = run_bounds[0] + x1
                run_x2 = run_bounds[1] + x1
                rod_center = ( run_x1 + run_x2 ) // 2
                rod_bounds = ( y1, rod_center, run_x2 - run_x1, run_x1, run_x2 )
                all_bounds.append(rod_bounds)

            # debug
            if self.compute_overlay:
                # display the curve at the bottom of the SR rect
                if y1 == sr.y + sr.h - 1:
                    self.draw_rect(sr,    (  0, 255, 0), width=1)
                    self.draw_curve(y_lu_u8 / 255 * sr.h, sr, (0, 255, 255))
                    self.draw_curve(selected_01 * sr.h, sr, (255, 0, 0))
                    self.draw_threshold(f_threshold / 255 * sr.h, sr, (0, 165, 255)) # orange
                    sr.move_by(0, sr.h)

        if len(all_bounds) < 2:
            return None

        # TBD: replace all_bounds array by 3 arrays for y/c/w.
        np_y = np.array( [ r[0] for r in all_bounds ] )
        np_c = np.array( [ r[1] for r in all_bounds ] )
        np_w = np.array( [ r[2] for r in all_bounds ] )
        # poly_c: deg=2 looked better than deg=3.
        poly_c = np.polynomial.Polynomial.fit(np_y, np_c, deg=2)
        poly_w = np.polynomial.Polynomial.fit(np_y, np_w, deg=1)

        return RodResult(
            frame_index=frame_index,
            initial_center=initial_rod_center,
            y_top=yt,
            y_bottom=yb,
            poly_c=poly_c,
            poly_w=poly_w,
        )

    def select_best_run(self, selected_01, min_size, max_size, rod_center, ideal_rod_w):
        # Return run_bounds(x1: int, x2: int) or None

        # Find transitions (0 to 1 and 1 to 0)
        # Prepend/Append 0 to handle runs at the very start or end
        padded_u8 = np.pad(selected_01, (1, 1), "constant", constant_values=0)
        diffs_u8 = np.diff(padded_u8)
        # diff computes 0->1 = 1 or 1->0 = 255 (-1 on an uint8 buffer)
        starts = np.where(diffs_u8 == 1)[0]
        ends = np.where(diffs_u8 == 255)[0]
        lengths = ends - starts

        # Identify which runs to keep
        valid_indices = np.where((lengths >= min_size) & (lengths <= max_size))[0]

        target_x1 = rod_center - ideal_rod_w / 2
        target_x2 = rod_center + ideal_rod_w / 2
        best_score = 0
        selected_bounds = None  # None or ( x1, x2 )

        # Select the one that is the closest to the desired rod center
        for idx in valid_indices:
            x1 = starts[idx]
            x2 = ends[idx]

            # the score is the IoU of this segment vs the target
            score = self.iou(x1, x2, target_x1, target_x2)
            if score > best_score:
                best_score = score
                selected_bounds = ( int(x1), int(x2) )

        return selected_bounds

    def iou(self, ax1, ax2, bx1, bx2):
        """Computes IoU (Intersection over Union) between 2 segments A and B"""
        intersection = max(0, min(ax2, bx2) - max(ax1, bx1))
        union = (ax2 - ax1) + (bx2 - bx1) - intersection
        return intersection / union if union > 0 else 0

    def export(self):
        print(f"@@ RodDetector export")
        rods =  [ v.to_json() for k, v in self.rods.items() ]
        return {
            "poly_rods": rods,
        }

    def read_json(self, data):
        if "poly_rods" in data:
            rods = [RodResult.from_json(r) for r in data["poly_rods"]]
            rods.sort(key=lambda r: r.frame_index)
            for r in rods:
                self.rods[r.frame_index] = r
            print(f"@@ RodDetector imported {len(rods)} rod results")
        self.fix_rods()

    def pre_release(self):
        self.fix_rods()
        super().pre_release()

    def fix_rods(self):
        if not self.rods:
            return
        if self.rods_fixed:
            return
        self.fix_bent_rods()
        self.fix_missing_rods()

    def fix_bent_rods(self):
        self.rods_fixed = True

        c2 = np.array( [ abs(r.poly_c.coef[2]) for r in self.rods.values() ] )
        c2_abs = np.median( c2 )
        threshold = 2.25 * c2_abs

        remove_frames = []

        for frame, rod in self.rods.items():
            c2 = rod.poly_c.coef[2]
            is_bent = abs(c2) > threshold
            if is_bent:
                remove_frames.append(frame)
                print(f"@@ [{frame:04d}] REMOVE {c2:.4f} > {threshold:.4f} --- {'THRESHOLD' if is_bent else '-'}")

        # Remove frames in place in the rods dictionary
        for frame in remove_frames:
            self.rods.pop(frame, None)
        print(f"@@ Removed {len(remove_frames)} bent rods; {len(self.rods)} rods left.")

    def fix_missing_rods(self):
        if not self.rods:
            return

        frames_existing = np.sort(np.array( [ r.frame_index for r in self.rods.values() ] ))
        # Find where the gap between consecutive frames is > 1
        diffs = np.diff(frames_existing)
        gap_indices = np.where(diffs > 1)[0]

        for idx in gap_indices:
            frame1 = frames_existing[idx]
            frame2 = frames_existing[idx + 1]

            rod1 = self.rods[frame1]
            rod2 = self.rods[frame2]

            # Indices we need to fill
            missing_range = np.arange(frame1 + 1, frame2)

            num_frames = frame2 - frame1

            ic1 = rod1.initial_center
            yt1 = rod1.y_top
            yb1 = rod1.y_bottom
            ic1_range = rod2.initial_center - ic1
            yt1_range = rod2.y_top - yt1
            yb1_range = rod2.y_bottom - yb1

            print(f"@@ Interpolating from {frame1} to {frame2}")

            for frame in missing_range:
                t = (frame - frame1) / num_frames

                ic = int(ic1 + t * ic1_range)
                yt = int(yt1 + t * yt1_range)
                yb = int(yb1 + t * yb1_range)

                result = RodResult(
                    frame_index=int(frame),
                    initial_center=ic,
                    y_top=yt,
                    y_bottom=yb,
                    poly_c=self.interpolate_polys(rod1.poly_c, rod2.poly_c, t),
                    poly_w=self.interpolate_polys(rod1.poly_w, rod2.poly_w, t),
                )

                self.rods[frame] = result
                print(f"@@ Interp Rod [{frame:04d}] {t:.3f} -> {result}")

        # Finally sort the dictionary by key to maintain a consistent frame ordering
        # (Python dicts are ordered so new keys were added at the end, we need them in key order)
        self.rods = dict(sorted(self.rods.items()))

    def interpolate_polys(self, poly_a, poly_b, t):
        """
        t: 0.0 is poly_a, 1.0 is poly_b
        """
        # Convert both to standard form to ensure coefficients
        # are in the same 'pixel' units
        pa_std = poly_a.convert(domain=poly_a.domain)
        pb_std = poly_b.convert(domain=poly_a.domain) # Match domains

        # 2. Linear interpolation of coefficients
        # (1-t)*A + t*B
        new_coef = (1 - t) * pa_std.coef + t * pb_std.coef

        # 3. Return a new Polynomial
        return np.polynomial.Polynomial(new_coef, domain=poly_a.domain)

# ~~
