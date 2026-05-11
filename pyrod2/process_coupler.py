import bisect
import cv2
import math
import numpy as np
from processor import ProcessorBase
from point import Point
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
        self.roi_rect = None
        self.tracker_templates = {}
        self.couplers = {}
        self.couplers_fixed = False

    def init_size(self, width, height):
        super().init_size(width, height)
        self.roi_center = width / 2

    def init_overlay(self, frame):
        super().init_overlay(frame)

    def filter(self, window_title, frame_index, frame):
        if ( (frame_index == self.start_frame or self.start_frame == 0)
                and self.couplers_fixed
                and len(self.tracker_templates) > 0
                and len(self.couplers) > 0 ):
            self.next_processor_requested = True
            return frame

        h, w = frame.shape[:2]
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lu = lab[:, :, 0]
        frame = cv2.cvtColor(lu, cv2.COLOR_GRAY2BGR)

        while self.current_template is None:
            print(f"@@ [frame #{frame_index:04d}] Select a valid top coupler area to continue.")
            self.current_template = self.select_roi(window_title, frame_index, frame, lu)
            if self.current_template:
                self.tracker_templates[frame_index] = self.current_template.copy()
                print(f"@@ [frame #{frame_index:04d}] coupler {self.current_template.rect}, center {self.current_template.rect.center()}")

        srect = self.get_search_window()

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
                result = CouplerResult(
                    frame_index = frame_index,
                    center = self.current_template.rect.centerPoint(),
                    quality = quality,
                    coupler_ref = self.current_template.frame_index,
                )
                self.couplers[frame_index] = result
                self.update_roi_center(result.center.x)
            else:
                self.update_roi_center(w / 2)

            if self.compute_overlay:
                self.draw_heatmap(srect, res, self.current_template)

            color = (0, 255, 255)  # debug search frame is yellow
        else:
            # Reuse previous result
            result = self.couplers[frame_index]
            self.update_roi_center(result.center.x)
            # we didn't record max_val and we just need it for debug display below.
            max_val = quality = result.quality
            # update the display rect for debug display
            curr_c = self.current_template.rect.center()
            new_c = result.center
            self.current_template.rect.move_by(new_c.x - curr_c[0], new_c.y - curr_c[1])
            color = (255, 255, 0)  # debug search frame is cyan

        if self.compute_overlay:
            self.draw_rect(srect, color)
            color = (0, 0, 255) if quality < QUALITY_THRESHOLD else (0, 255, 0)
            self.draw_rect(self.current_template.rect, color)
            text1 = f"{max_val:4.2f} : {quality:4.2f}"
            texty = srect.y + srect.h - int(quality * srect.h)
            color = (0, 0, 255) if quality < QUALITY_THRESHOLD else (0, 165, 255) # orange
            cv2.putText(self.overlay, text1,
                    (srect.x, texty),           # bottom-left coord
                    cv2.FONT_HERSHEY_DUPLEX,    # font
                    1,                          # font scale
                    color,                      # color
                    1 )                         # line thickness
            # print(f"@@ [frame #{frame_index}] --> val {max_val} at track {self.current_template.rect}")

        return frame

    def select_roi(self, window_info, frame_index, frame, lu):
        """Returns None if no ROI selected, otherwise returns a Template tuple."""
        super().select_roi_called()
        # print(f"@@ [frame #{frame_index}] Select top coupler area. 'c' to cancel, space/return to accept.")
        window_info["pre_display"]()
        rect = cv2.selectROI(window_info["title"], frame, showCrosshair=True, fromCenter=False)
        window_info["post_display"]()
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

    def draw_heatmap(self, srect, res, template):
        # 1. Clip and Scale
        # Since we care about the match (near 1.0), we can clip negative values to 0
        # and scale the 0-1.0 range to 0-255.
        res_clipped = np.clip(res, 0, 1.0)
        res_u8 = (res_clipped * 255).astype(np.uint8)

        # 2. Apply Colormap
        # COLORMAP_JET: 255 (1.0) -> Red, 0 (0.0) -> Blue
        heatmap = cv2.applyColorMap(res_u8, cv2.COLORMAP_JET)
        x = srect.x + template.rect.w // 2
        y = srect.y + template.rect.h // 2
        h, w = heatmap.shape[:2]
        dest = self.overlay[y : y + h, x : x + w]
        mask = res_u8 > 64
        dest[mask] = heatmap[mask]

    def update_roi_center(self, center_x):
        self.roi_center = self.roi_center * 0.9 + center_x * 0.1

    def get_search_window(self):
        width = self.width
        height = self.height
        template_rect = self.current_template.rect
        w = int(ROI_WIDTH_PCT * width)
        if self.roi_rect is None:
            y = template_rect.y - template_rect.h
        else:
            y = self.roi_rect.y
        x = int(self.roi_center - w // 2)
        h = height - y
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
            print(f"@@ CouplerTracker imported {len(self.tracker_templates)} coupler templates")
            if self.current_template is None and self.tracker_templates:
                indices = sorted(self.tracker_templates.keys())
                idx = bisect.bisect_right(indices, self.start_frame)
                if len(indices) == 1 or idx == 0:
                    self.current_template = self.tracker_templates[indices[0]].copy()
                else:
                    self.current_template = self.tracker_templates[indices[idx - 1]].copy()
        if "couplers" in data:
            couplers = [CouplerResult.from_json(c) for c in data["couplers"]]
            couplers.sort(key=lambda c: c.frame_index)
            for c in couplers:
                self.couplers[c.frame_index] = c
            print(f"@@ CouplerTracker imported {len(couplers)} coupler results")
        self.fix_coupler_movement()

    def pre_release(self):
        self.fix_coupler_movement()
        super().pre_release()

    def release(self):
        super().release()

    def fix_coupler_movement(self):
        if not self.couplers:
            return
        if not self.couplers_fixed:
            self.fix_y()
            self.fix_x()
            self.fix_missing_frames()
            self.couplers_fixed = True

    def fix_y(self):
        # 1- Filter on Y first.

        # By design, the coupler is generally at the same height, within a certain window.
        # This computes the median position and a threshold outside of the typical position.
        # We then fix just the values outside the median's threshold.

        # tuples (0=frame_index, 1=y)
        data = [ (f, c.center.y ) for f, c in self.couplers.items() ]
        values = np.array([ d[1] for d in data ])

        # Using Median Absolute Deviation
        median = np.median(values)
        median_abs_dev = np.median(np.abs(values - median))  # median jitter around median Y
        # threshold = median + (3 * 1.4826 * median_abs_dev)
        threshold = 5 * median_abs_dev
        is_jerk = np.abs(values - median) > threshold

        # Interpolate the incorect Y positions
        np_indices = np.arange(len(values))
        fixed = np.interp(np_indices, np_indices[~is_jerk], values[~is_jerk])

        last_f = 0
        for idx, d in enumerate(data):
            f, oy = d
            # DEBUG print
            # if f != last_f + 1:
            #     print("-----------------")
            ny = fixed[idx]
            jrk = is_jerk[idx]
            if jrk:
                # DEBUG print
                # print(f"@@ [{f:04d}] y: {oy:4d} --> {ny:6.2f} ,  {'**** JRK ****' if jrk else '-'}")
                # fix the Y in the couplers data set
                self.couplers[f].center.move_by(0, int(ny - oy))
            last_f = f
        print(f"@@ Delta Y median: {median}, threshold: {threshold}")

    def fix_x(self):
        # 2- Filter on X next
        # # The coupler moves around in X so instead of cleaning the absolute X
        # # value, we use an average window combined with a median absolute deviation.

        # tuples (0=frame_index, 1=x)
        data = [ (f, c.center.x ) for f, c in self.couplers.items() ]
        values = np.array([ d[1] for d in data ])
        fixed = self.hampel_filter_numpy(values)

        last_f = 0
        for idx, d in enumerate(data):
            f, ox = d
            # DEBUG print
            # if f != last_f + 1:
            #     print("-----------------")
            nx = int(fixed[idx])
            jrk = nx != ox
            if jrk:
                # DEBUG print
                # real_jrk = abs(nx - ox) > 1
                # print(f"@@ [{f:04d}] x: {ox:4d} --> {nx:4d} ,  {'**** X-JRK ****' if real_jrk else '-'}")
                # fix the X in the couplers data set
                self.couplers[f].center.move_by(nx - ox, 0)
            last_f = f

    def hampel_filter_numpy(self, x, window_size=10, n_sigmas=2.5):
        """
        x: 1D numpy array (X positions)
        window_size: Look-ahead/look-back distance (total window is 2*window_size + 1)
        n_sigmas: Sensitivity (lower is more aggressive)

        Note: if outliers seemed to not be cleaned, that typically means the threshold is
        too high -- try lowering n_sigmas in this case. A smaller window also means a more
        localized median, which can help make smoothing more aggresive.
        """
        n = len(x)
        new_x = x.copy()
        k = 1.4826 # Scale factor for Gaussian distribution

        if n < window_size:
            return new_x

        # Create a sliding window view
        # This creates a virtual (N, window_len) array without copying memory
        pad = window_size
        full_window = 2 * window_size + 1
        views = np.lib.stride_tricks.sliding_window_view(x, full_window)

        # Calculate local medians and MADs (Median Absolute Deviations)
        local_medians = np.median(views, axis=1)
        local_mads = k * np.median(np.abs(views - local_medians[:, None]), axis=1)

        # We need to pad the results because sliding_window reduces the array size
        # We'll pad with the original values for the edges
        diff = np.abs(x[pad:-pad] - local_medians)
        thresholds = n_sigmas * local_mads
        outlier_mask = diff > thresholds
        print(f"@@ Delta X median: {np.median(local_medians)}, threshold: {np.median(thresholds):4.2f}, {np.count_nonzero(outlier_mask)} outliers")

        # Replace outliers with local median
        # We offset by 'pad' because 'outlier_mask' corresponds to the center of the windows
        indices = np.where(outlier_mask)[0] + pad
        new_x[indices] = local_medians[outlier_mask]

        return new_x

    def fix_missing_frames(self):
        frames_existing = np.array([c.frame_index for c in self.couplers.values()])
        x_existing = np.array([c.center.x for c in self.couplers.values()])
        y_existing = np.array([c.center.y for c in self.couplers.values()])

        # Full range of frames from first to last
        all_frames = np.arange(frames_existing.min(), frames_existing.max() + 1)

        # Find which frames are actually missing
        missing_mask = np.isin(all_frames, frames_existing, invert=True)
        missing_frames = all_frames[missing_mask]
        print(f"@@ {len(missing_frames)} missing frames to interpolate")

        # Use np.interp to find the X and Y for all missing frames at once
        # Syntax is np.interp(target_x, known_x, known_y)
        interp_x = np.interp(missing_frames, frames_existing, x_existing)
        interp_y = np.interp(missing_frames, frames_existing, y_existing)

        # Build a list, similar to bisect_right mapping all misisng frames numbers to the
        # rightmost existing frame index. That's really an insertion index which we convert
        # below to a frame_index number.
        insertion_indices = np.searchsorted(frames_existing, missing_frames, side='right')

        # Use zip to read all the input arrays at the same time (they have the same length)
        for ins, f, x, y in zip(insertion_indices, missing_frames, interp_x, interp_y):
            previous_f_idx = frames_existing[ins - 1] if ins > 0 else frames_existing[0]
            previous_f = self.couplers[previous_f_idx]
            x = int(x)
            y = int(y)
            f = int(f)
            self.couplers[f] = CouplerResult(
                frame_index = f,
                center = Point(x, y),
                quality = 0,  # remember it was a missing frame of low quality
                coupler_ref = previous_f.coupler_ref,
            )
            print(f"@@ Interp Coupler [{previous_f_idx:04d}] -> {self.couplers[f]}")

        # Finally sort the dictionary by key to maintain a consistent frame ordering
        # (Python dicts are ordered so new keys were added at the end, we need them in key order)
        self.couplers = dict(sorted(self.couplers.items()))
