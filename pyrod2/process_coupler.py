import bisect
import cv2
import math
import numpy as np
from processor import ProcessorBase
from rect import Rect
from coupler_result import CouplerResult
from coupler_template import CouplerTemplate

ROI_WIDTH_PCT = 1/3
QUALITY_THRESHOLD = 0.1

class CouplerTracker(ProcessorBase):
    def __init__(self, start_frame):
        super().__init__()
        self.start_frame = start_frame
        self.current_template = None
        self.current_search_rect = None
        self.tracker_templates = {}
        self.couplers = {}

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

        srect = self.current_search_rect
        if srect == None:
            srect = self.current_search_rect = self.get_search_window(w, h)

        has_result = frame_index in self.couplers

        if not has_result:
            search_lu = lu[srect.y : srect.y + srect.h, srect.x : srect.x + srect.w]
            res = cv2.matchTemplate(search_lu, self.current_template.template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            med_lu = np.median(search_lu)
            quality = max_val * med_lu / 255

            if quality >= QUALITY_THRESHOLD:
                # Update current template with best match and add result
                self.current_template.rect.x = max_loc[0] + srect.x
                self.current_template.rect.y = max_loc[1] + srect.y
                self.couplers[frame_index] = CouplerResult(
                    frame_index = frame_index,
                    quality = quality,
                    center = self.current_template.rect.centerPoint(),
                    coupler_ref = self.current_template.frame_index,
                )
            color = (0, 255, 255)  # debug search frame is yellow
        else:
            # Reuse previous result
            result = self.couplers[frame_index]
            # we didn't record max_val and we just need it for debug display below.
            max_val = quality = result.quality
            # update the display rect for debug display
            curr_c = self.current_template.rect.center()
            new_c = result.center
            self.current_template.rect.moveBy(new_c.x - curr_c[0], new_c.y - curr_c[1])
            color = (255, 255, 0)  # debug search frame is cyan

        if self.compute_overlay:
            self.draw_rect(srect, color)
            color = (0, 0, 255) if quality < QUALITY_THRESHOLD else (0, 255, 0)
            self.draw_rect(self.current_template.rect, color)
            text1 = f"{max_val:4.2f} : {quality:4.2f}"
            texty = srect.y + srect.h - int(quality * srect.h)
            color = (0, 0, 255) if quality < QUALITY_THRESHOLD else (0, 165, 255)
            cv2.putText(self.overlay, text1,
                    (srect.x, texty),           # bottom-left coord
                    cv2.FONT_HERSHEY_DUPLEX,    # font
                    .75,                        # font scale
                    color,                      # color
                    1 )                         # line thickness
            # print(f"@@ [frame #{frame_index}] --> val {max_val} at track {self.current_template.rect}")

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
            return CouplerTemplate(
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

    def export(self):
        print(f"@@ CouplerTracker export")
        couplers =  [ v.to_json() for k, v in self.couplers.items() ]
        templates = [ v.to_json() for k, v in self.tracker_templates.items() ]
        return {
            "coupler_templates": templates,
            "couplers": couplers,
        }

    def read_json(self, data):
        if "coupler_templates" in data:
            for t in data["coupler_templates"]:
                template = CouplerTemplate.from_json(t)
                self.tracker_templates[template.frame_index] = template
            if self.current_template == None and self.tracker_templates:
                indices = sorted(self.tracker_templates.keys())
                idx = bisect.bisect_right(indices, self.start_frame)
                if len(indices) == 1 or idx == 0:
                    self.current_template = self.tracker_templates[indices[0]]
                else:
                    self.current_template = self.tracker_templates[indices[idx - 1]]
        if "couplers" in data:
            couplers = [CouplerResult.from_json(c) for c in data["couplers"]]
            couplers.sort(key=lambda c: c.frame_index)
            for c in couplers:
                self.couplers[c.frame_index] = c
        self.fix_coupler_movement()

    def release(self):
        super().release()

    def fix_coupler_movement(self):
        last_center = None
        data = [] # tuples (0=frame_index, 1=dx, 2=y)
        for f, c in self.couplers.items():
            center = c.center
            if last_center is not None:
                delta = last_center.delta_to(center)
                dx = abs(delta[0])
                dy = abs(delta[1])
                data.append( (f, dx, center.y ) )
            last_center = center

        # Filter on Y first.
        all_y = np.array([ d[2] for d in data ])
        # deltas_m = np.array([ d[3] for d in data ])

        # # Using Interquartile Range (IQR)
        # q1, q3 = np.percentile(deltas_y, [25, 75])
        # iqr = q3 - q1
        # threshold = q3 + (1.5 * iqr)
        # is_jerk = deltas_y > threshold

        # Using Median Absolute Deviation
        median = np.median(all_y)
        median_abs_dev = np.median(np.abs(all_y - median))  # median jitter around median Y
        # threshold = median + (3 * 1.4826 * median_abs_dev)
        threshold = 5 * median_abs_dev
        is_jerk = np.abs(all_y - median) > threshold

        # Interpolate the incorect Y positions
        np_indices = np.arange(len(all_y))
        all_y_fixed = np.interp(np_indices, np_indices[~is_jerk], all_y[~is_jerk])

        last_f = 0
        for idx, d in enumerate(data):
            f = d[0]
            if f != last_f + 1:
                print("-----------------")
            dx = d[1]
            oy = d[2]
            ny = all_y_fixed[idx]
            mag = d[3]
            jrk = is_jerk[idx]
            print(f"@@ [{f:04d}] dx: {dx:4d}, y: {oy:4d} --> {ny:6.2f} , mag: {mag:4d}, {'**** JRK ****' if jrk else '-'}")
            last_f = f
        print(f"@@ Delta Y median: {median}, threshold: {threshold}")
