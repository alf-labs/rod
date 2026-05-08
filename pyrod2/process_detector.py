import cv2
import math
import numpy as np
from processor import ProcessorBase
from point import Point
from rect import Rect
from process_coupler import ROI_WIDTH_PCT, QUALITY_THRESHOLD
from rod_result import RodResult

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
        self.rods = {}
        self.rods_fixed = False

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
            srect = self.get_search_window(w, h, coupler_template)
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
            run_color = (0, 165, 255)
            first_lc = None
            last_lc = None
            for y1 in range(result.y_top, result.y_bottom):
                lc = result.poly_c(y1)
                if first_lc is None: first_lc = lc
                last_lc = lc
                lw = result.poly_w(y1)
                lx1 = int(lc - lw / 2)
                lx2 = int(lc + lw / 2)
                cv2.line(self.overlay, (lx1, y1), (lx2, y1), run_color, 1)
            # print(f"@@ DEBUG [{frame_index:04d} LC {first_lc:.3f} -> {last_lc:.3f}]")

        # # For debug purposes, we display any changes made to the LU image.
        # frame = cv2.cvtColor(lu, cv2.COLOR_GRAY2BGR)
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
                # display current run
                # if run_bounds is not None:
                #     run_color = (0, 165, 255)
                #     lx1 = rod_bounds[3]
                #     lx2 = rod_bounds[4]
                #     cv2.line(self.overlay, (lx1, y1), (lx2, y1), run_color, 1)
                # display the curve at the bottom of the SR rect
                if y1 == sr.y + sr.h - 1:
                    self.draw_rect(sr,    (  0, 255, 0), width=1)
                    self.draw_curve(y_lu_u8 / 255 * sr.h, sr, (0, 255, 255))
                    self.draw_curve(selected_01 * sr.h, sr, (255, 0, 0))
                    self.draw_threshold(f_threshold / 255 * sr.h, sr, (0, 165, 255)) # orange
                    sr.move_by(0, sr.h)

        if len(all_bounds) < 2:
            return None

        # x1s = [r[3] for r in all_bounds]
        # x2s = [r[4] for r in all_bounds]
        # dx1 = max(x1s) - min(x1s)
        # dx2 = max(x2s) - min(x2s)
        # print(f"@@ [{frame_index:04d}] max1:{dx1:3d}, {dx1 / (yb - yt)}")
        # print(f"@@ [{frame_index:04d}] max2:{dx2:3d}, {dx2 / (yb - yt)}")

        # TBD: replace all_bounds array by 3 arrays for y/c/w.
        np_y = np.array( [ r[0] for r in all_bounds ] )
        np_c = np.array( [ r[1] for r in all_bounds ] )
        np_w = np.array( [ r[2] for r in all_bounds ] )
        poly_c = np.polynomial.Polynomial.fit(np_y, np_c, deg=2)    # 2 or 3?
        poly_w = np.polynomial.Polynomial.fit(np_y, np_w, deg=1)    # 1 or 2?

        return RodResult(
            frame_index,
            initial_rod_center,
            yt, yb,
            poly_c, poly_w,
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
        self.fix_rod_movement()

    def pre_release(self):
        self.fix_rod_movement()
        super().pre_release()

    def fix_rod_movement(self):
        if not self.rods:
            return
        if self.rods_fixed:
            return
        self.rods_fixed = True
        # last_top = None
        # last_frame = None
        # ts = []
        # bs = []

        c2 = np.array( [ abs(r.poly_c.coef[2]) for _, r in self.rods.items() ] )
        c2_abs = np.median( c2 )
        threshold = 2.25 * c2_abs

        remove_frames = []

        for frame, rod in self.rods.items():
            c2 = rod.poly_c.coef[2]
            is_bent = abs(c2) > threshold
            if is_bent:
                remove_frames.append(frame)
                print(f"@@ [{frame:04d}] REMOVE {c2:.4f} > {threshold:.4f} --- {'THRESHOLD' if is_bent else '-'}")

            # x2.append(c2)
            # print(f"@@ [{frame:04d}] {c2:.4f} ==> {score1:.4f} vs {score2:.4f} vs {score3:.4f} --- {'THRESHOLD' if is_bent else '-'}")
            # y_top = rod.y_top
            # y_bot = rod.y_bottom
            # y_mid = (y_top + y_bot) / 2

            # x_top = rod.poly_c(y_top)
            # x_mid = rod.poly_c(y_mid)
            # x_bot = rod.poly_c(y_bot)



            # if last_top is not None:
            #     dx_top = (x_top - last_top)
            #     dx_bot = (x_bot - x_top)
            #     print(f"@@ [{frame:04d}] {x_top:.1f} x {rod.y_top:03d} > {x_bot:.1f} x {rod.y_bottom:03d} -- DX Top: {dx_top:.3f}, Bot: {dx_bot:.3f}")
            #     ts.append(dx_top)
            #     bs.append(dx_bot)

            # last_top = x_top
            # last_frame = frame

        # Remove frames in place in the rods dictionary
        for frame in remove_frames:
            self.rods.pop(frame, None)
        print(f"@@ Removed {len(remove_frames)} bent rods; {len(self.rods)} rods left.")

        # print(f"@@ COEF 2 MEDIAN Global {c2_med:.3f}, abs {c2_abs:.3f}, abs corrected {abs_med:.3f} -- thresold {threshold:.3f}")

        # x2s = np.array( x2 )
        # a2s = np.abs( x2s )
        # print(f"@@ COEF 2 MIN T {np.min(x2s):.3f}, MEDIAN {np.median(x2s):.3f}, MAX {np.max(x2s):.3f}")
        # print(f"@@ COEF 2 MIN T {np.min(a2s):.3f}, MEDIAN {np.median(a2s):.3f}, MAX {np.max(a2s):.3f}")

        # ts = np.array( ts )
        # bs = np.array( bs )
        # print(f"@@ MIN T {np.min(ts):.3f}, B {np.min(bs):.3f}")
        # print(f"@@ MAX T {np.max(ts):.3f}, B {np.max(bs):.3f}")
        # print(f"@@ MED T {np.median(ts):.3f}, B {np.median(bs):.3f}")

# ~~
