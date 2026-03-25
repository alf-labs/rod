#!/usr/bin/python
# Ensure we're running from the Virtual Env version
import os
if not "VIRTUAL_ENV" in os.environ:
    print("ERROR: Run this from venv using 'source ./venv_catd/bin/activate' first")
    exit(1)
IS_RPI = os.path.isfile("/etc/rpi-issue")

import base64
import sys
import time

try:
    import cv2
    import numpy as np
    import imutils
    import scipy
    from flask import Flask, render_template, Response, request, jsonify
except ModuleNotFoundError as e:
    print(f"ERROR: Missing library. {e}")
    print( "To fix: $ pip install opencv-python numpy scipy imutils flask")
    print(f"or    : $ python {sys.argv[0]}")
    exit(1)

WINDOW_TITLE = "Rod Sample"
VIDEOS = [
    "../samples/rod1_front_randall_up_2025-03-23.mp4",
    "../samples/rod1_rear_randall_up_2025-03-23.mp4",
]

FPS = 30
FPS_MS = 1000//FPS
GRAPH_Y_OFFSET = 10
NUM_BOTTOM_ROWS_CV_PCT = 50/720
ROD_WIDTH = 35/1280
ROD_W_RANGE = (25/1280, 40/1280)

def draw_line(source, channel_index, y, color, dest):
    if source.ndim == 3:
        channel = source[:, :, channel_index]
        cvalue = lambda x: int(channel[y, x])
    elif source.ndim == 2:
        cvalue = lambda x: int(source[y, x])
    else:
        cvalue = lambda x: int(source[x])

    h, w = dest.shape[:2]
    if y < 0:
        y = h + y

    lx = 0
    ly = cvalue(0)
    dy = h - GRAPH_Y_OFFSET
    for x in range(1, w):
        curr = cvalue(x)
        cv2.line(dest, (lx, dy - ly), (x, dy - curr), color, 2)
        lx = x
        ly = curr


class Rod:
    def __init__(self, left, right, score):
        self.left = left
        self.right = right
        self.score = score

    def center(self):
        return (self.left + self.right) / 2

    def width(self):
        return self.right - self.left

    def __repr__(self):
        return f"Rod( {self.left:.3f} -> {self.right:.3f} ; width {self.width():.3f} ; score {self.score:.3f} )"



class DetectorBase:
    def __init__(self):
        # Overlay is (B,G,R)
        self.overlay = None
        self.trigger_pause = False

    def init_size(self, width, height):
        self.width = width
        self.height = height

    def init_overlay(self, frame):
        if self.overlay is None:
            self.overlay = np.zeros_like(frame)
        else:
            self.overlay[:] = (0, 0, 0)

    def combine_overlay(self, src_dst):
        gray_overlay = cv2.cvtColor(self.overlay, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_overlay, 1, 255, cv2.THRESH_BINARY)
        src_dst[mask > 0] = self.overlay[mask > 0]
        return src_dst

    def filter(self, frame):
        return frame


class Detector2(DetectorBase):
    def __init__(self):
        super().__init__()
        self.last_cv_lu = None
        self.last_threshold = None
        self.current_rod = None
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))


    def init_size(self, width, height):
        super().init_size(width, height)

        # Number of rows to scan at the bottom to get the vertical per-band CV
        self.num_bottom_rows_cv = int(NUM_BOTTOM_ROWS_CV_PCT * height)

        # Width, as a fraction of the screen width
        self.rod_width_px = int(ROD_WIDTH * width)
        self.rod_w_range_px = ( int(ROD_W_RANGE[0] * width), int(ROD_W_RANGE[1] * width) )
        print("Rod Width PX: ", self.rod_width_px, "in range", self.rod_w_range_px)

    def weight(self, a, b, weight_a=0.75):
        return a * weight_a + b * (1 - weight_a)

    def weight_asymetric(self, a, b, weight_a_up=0.25, weight_a_down=0.75):
        if b > a:   # going up
            return a * weight_a_up + b * (1 - weight_a_up)
        else:
            return a * weight_a_down + b * (1 - weight_a_down)

    def y_np_scalar(self, np_scalar, top=0):
        if top > 0:
            np_scalar -= top
            np_scalar *= 1000
            np_scalar += 255
            return np.clip(np_scalar, a_min=None, a_max=255).item()
        else:
            return np.clip(np_scalar * 1000, a_min=None, a_max=255).item()

    def y_np_vector(self, np_vect):
        return np.clip(np_vect * 1000, a_min=None, a_max=255)

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

    def find_rod_peaks(self, cv_peaks):
        if self.current_rod is None:
            score_center = cv_peaks.size / 2
            rod_left = -1
            rod_right = -1
        else:
            score_center = self.current_rod.center()
            rod_left = self.current_rod.left
            rod_right = self.current_rod.right

        peaks, props = scipy.signal.find_peaks(
            cv_peaks,
            prominence=0.50, # Minimum 'depth' of the valley to be considered
            # width=self.rod_w_range_px,    # Looking for our ~30px rod
            # rel_height=0.5     # Calculate width at 50% of the prominence
        )
        # print("@@ peaks", peaks, " // props", props)

        candidates = []

        y = self.height - GRAPH_Y_OFFSET

        rod_width = self.rod_width_px
        min_width = self.rod_w_range_px[0]
        max_width = self.rod_w_range_px[1]

        left_bases = props["left_bases"]
        right_bases = props["right_bases"]
        num_peaks = len(peaks)

        for i in range(0, num_peaks):
            left_px = left_bases[i]
            right_px = right_bases[i]
            peak = peaks[i]
            peak_cv = cv_peaks[peak].item()
            midpoint = (left_px + right_px) / 2
            width = right_px - left_px

            # candidates.append(
            #     f"<peak {peak} = {peak_cv}, {left_px} < {midpoint} < {right_px}, w/ {width} >"
            # )

            # Apply Width Constraints
            # yet check any peak overlapping the current rod
            cond_width = min_width <= width <= max_width
            cond_middle = rod_left <= peak <= rod_right
            if cond_width or cond_middle:

                # Calculate Center Score.
                # Score starts with the peak's CV value, so we want the highest one.
                # It is degraded (lowered) by the distance from the center,
                # and further degraded by the absolute difference from the ideal width.
                delta_center = peak - score_center
                delta_width = width - rod_width
                score = peak_cv - abs(delta_center) - abs(delta_width) / 100

                # if cond_middle:
                #     # We want to mostly use the old rod position, slightly shifted towards
                #     # the new midpoint; we're trying to keep the same width.
                #     left_px = int(self.weight(rod_left, rod_left + delta_center))
                #     right_px = int(self.weight(rod_right, rod_right + delta_center))

                candidates.append( Rod(left_px, right_px, score) )

                # ys = int(y + max(0, min(score + 255 - 1, 255)))
                ys = y - 255
                y += 5
                cv2.line(self.overlay, (left_px, ys), (right_px, ys), (255, 0, 0), 3)

        # Find best match (highest score)
        if not candidates:
            return None
        # print(candidates)
        # print("@@ ", self.last_threshold, " >> ", candidates)
        best_candidate = max(candidates, key=lambda x: x.score)

        # Ignore the best candidate if its score is drastically worse than the current one.
        # Since the score is a number of pixels off the center of the current rod, we can
        # compare the score delta to the rod width.
        if self.current_rod is not None:
            curr_score = self.current_rod.score
            if best_candidate.score < curr_score - 2 * rod_width:
                return None

        return best_candidate


    def find_rod_valleys(self, cv_under_threshold):
        """
        Finds the rod by searching for low-variance valleys in a 1D signal.
        """
        if self.current_rod is None:
            score_center = cv_under_threshold.size / 2
            rod_left = -1
            rod_right = -1
        else:
            score_center = self.current_rod.center()
            rod_left = self.current_rod.left
            rod_right = self.current_rod.right

        # Group contiguous 'valley' pixels
        # labels is an array where each valley is numbered 1, 2, 3...
        labels, num_features = scipy.ndimage.label(cv_under_threshold)

        candidates = []

        y = self.height - GRAPH_Y_OFFSET

        rod_width = self.rod_width_px
        min_width = self.rod_w_range_px[0]
        max_width = self.rod_w_range_px[1]


        for i in range(1, num_features + 1):
            indices = np.where(labels == i)[0]
            width = len(indices)
            left_px = indices[0].item()
            right_px = indices[-1].item()
            midpoint = (left_px + right_px) / 2

            # Apply Width Constraints
            # yet check any segment overlapping the current rod
            cond_width = min_width <= width <= max_width
            cond_middle = rod_left <= midpoint <= rod_right
            if cond_width or cond_middle:

                # Calculate Center Score
                # Lower distance to center = smaller score -- we want the lowest score
                delta_center = midpoint - score_center
                score = abs(delta_center)
                # Score is also degraded by how much width differs from expected width
                score += abs(width - rod_width) / 10

                if cond_middle:
                    # We want to mostly use the old rod position, slightly shifted towards
                    # the new midpoint; we're trying to keep the same width.
                    left_px = int(self.weight(rod_left, rod_left + delta_center))
                    right_px = int(self.weight(rod_right, rod_right + delta_center))

                candidates.append( Rod(left_px, right_px, score) )

                ys = int(y - min(score / 2, 255))
                cv2.line(self.overlay, (left_px, ys), (right_px, ys), (255, 0, 0), 3)

        # Find best match (lowest score)
        if not candidates:
            return None
        # print("@@ ", self.last_threshold, " >> ", candidates)
        best_candidate = min(candidates, key=lambda x: x.score)

        # Ignore the best candidate if its score is drastically worse than the current one.
        # Since the score is a number of pixels off the center of the current rod, we can
        # compare the score delta to the rod width.
        if self.current_rod is not None:
            curr_score = self.current_rod.score
            if best_candidate.score > curr_score + 2 * rod_width:
                return None

        return best_candidate

    def merge_rod(self, new_rod):
        if self.current_rod is None:
            self.current_rod = new_rod
        else:
            old = self.current_rod
            new_center = new_rod.center()
            old_center = old.center()

            # # Ignore new rod if it has moved by more than N rod widths
            # # For testing: we trigger a pause
            # delta_center = abs(new_center - old_center)
            # delta_threshold = 3 * self.rod_width_px
            # if delta_center > delta_threshold:
            #     self.trigger_pause = True

            new_rod = Rod(
                left=self.weight(old.left, new_rod.left),
                right=self.weight(old.right, new_rod.right),
                score=self.weight(old.score, new_rod.score)
            )
            self.current_rod = new_rod
            # print("@@ new rod:", new_rod)
            # print("@@ delta", delta_center, "<", delta_threshold, " @@ ", old, " >>> ", self.current_rod)
            # else:
            #     print("@@ delta", delta_center, ">=", delta_threshold)
        return self.current_rod

    def draw_rod(self, rod):
        left_px = int(rod.left)
        right_px = int(rod.right)
        y1 = self.height - GRAPH_Y_OFFSET
        y2 = y1 - 128
        cv2.rectangle(self.overlay, (left_px, y1), (right_px, y2), (0, 255, 0), 4)

    def draw_threshold(self, threshold_y, color_threshold, dest):
        y = self.height - int(threshold_y) - GRAPH_Y_OFFSET
        cv2.line(dest, (0, y), (self.width, y), color_threshold, 1)

    def extract_roi_for_cv(self, lu):
        # Extract the N bottom rows
        N=self.num_bottom_rows_cv
        bottom_lu = lu[-N:, :].copy()

        # Apply CLAHE to amplify local texture detail
        # We use a slightly lower clipLimit to avoid amplifying sensor noise too much
        lu_clahe = self.clahe.apply(bottom_lu)

        # Apply Histogram Stretching (Min-Max Normalization)
        # This stretches the resulting L channel to the full 0-255 range
        bottom_lu = cv2.normalize(lu_clahe, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        # zero the left and right areas we don't want to analyze
        q = self.width // 4
        bottom_lu[:, :q] = 0
        bottom_lu[:, -q:] = 0

        # for debugging, place the modified lu back into the original
        lu[-N:, :] = bottom_lu

        return bottom_lu

    def filter(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        # lu, au, bu = cv2.split(lab)     # uint8
        lu = lab[:, :, 0]

        # Extract the N bottom rows and zero the left and right areas we don't want to analyze
        bottom_lu = self.extract_roi_for_cv(lu)

        # Even though we only filter on the middle of the image, we keep a vector of self.width
        # for ease and consistency. Our images are not very large so it's not a big penalty.
        cv_lu = self.get_cv_vectorized(bottom_lu)

        # Smooth the CV vector
        window = 5
        cv_lu = np.convolve(cv_lu, np.ones(window)/window, mode="same")

        if self.last_cv_lu is not None:
            cv_lu = self.weight(cv_lu, self.last_cv_lu)
        self.last_cv_lu = cv_lu
        draw_line(self.y_np_vector(cv_lu), 0, -1, (0, 165, 255), self.overlay)

        # Adaptive thresholding
        epsilon = 1e-6
        cv_filtered = cv_lu[cv_lu > epsilon]
        if cv_filtered.size > 0:
            threshold = np.percentile(cv_filtered, self.rod_width_px)
        else:
            threshold = 0

        if self.last_threshold is not None:
            threshold = self.weight_asymetric(self.last_threshold, threshold)
        self.last_threshold = threshold

        # print(f"CVs: min: {np.min(cv_lu):.3f}, mean: {np.mean(cv_lu):.3f}, max: {np.max(cv_lu):.3f}, threshold: {threshold:.3f}")

        # self.detect_rod_prominence(cv_lu)

        cv_mask = cv_lu < threshold
        cv_peaks = (1 - cv_lu) * cv_mask
        peak_threshold = 1 - threshold
        draw_line(cv_peaks * 255, 0, -1, (0, 255, 255), self.overlay)
        self.draw_threshold(self.y_np_scalar(peak_threshold, 1), (0, 255, 0), self.overlay)
        # print("@@ peak_threshold: ", peak_threshold)

        # new_rod = self.find_rod_valleys(cv_mask)
        new_rod = self.find_rod_peaks(cv_peaks)
        if new_rod is not None:
            self.merge_rod(new_rod)
            self.draw_rod(self.current_rod)

        return cv2.cvtColor(lu, cv2.COLOR_GRAY2BGR)


class Main:
    def __init__(self):
        self.mx = 0
        self.my = 0
        self.zoom = 1
        self.view_org = False
        self.skip_num = 1

    def print_mouse_rgb(self, source, dest):
        x = self.mx
        y = self.my
        z = self.zoom
        b,g,r = source[y, x]
        text = f"X: {x}, Y: {y} | R: {r} G: {g} B: {b}"
        # cv2.rectangle(frame, (x + 10, y - 30), (x + 300, y), (0, 0, 0), -1)
        cv2.putText(dest, text,
            (10 * z, 20 * z),           # bottom-left coord
            cv2.FONT_HERSHEY_SIMPLEX,   # font
            z / 2,             # font scale
            (255, 255, 255),            # color
            z * 2)                  # line thickness

    def print_fps(self, loop_s, dest):
        fps = 1/loop_s if loop_s > 0 else 0
        ms = int(loop_s * 1000)
        text = f"{self.mx:03d} x, {ms} ms, {fps:.2f} fps"
        z = self.zoom
        cv2.putText(dest, text,
            (10 * z, 30 * z),           # bottom-left coord
            cv2.FONT_HERSHEY_DUPLEX,    # font
            z,                          # font scale
            (0, 255, 255),              # color
            z )                         # line thickness

    def run(self):
        print("@@ Run")

        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TITLE, 1920//2, 1080//2)
        def _mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_MOUSEMOVE:
                self.mx = x
                self.my = y
        cv2.setMouseCallback(WINDOW_TITLE, _mouse_callback)

        width = 0
        height = 0
        loop_s = 0
        init_once = True
        frame_count = 0
        paused = False

        detector = Detector2()

        cap = cv2.VideoCapture(VIDEOS[0])
        try:
            last_frame = None
            while cap.isOpened():
                start_loop_s = time.perf_counter()
                if paused:
                    frame = last_frame.copy()
                else:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    last_frame = frame.copy()

                frame_count += 1
                _skip_num = self.skip_num
                if _skip_num > 1:
                    if frame_count % _skip_num != 0:
                        continue

                if init_once:
                    height, width = frame.shape[:2]
                    print(f"Video size: {width}x{height}")
                    detector.init_size(width, height)
                    init_once = False

                detector.init_overlay(frame)
                result = detector.filter(frame)

                #self.print_mouse_rgb(frame, detector.overlay)
                self.print_fps(loop_s, detector.overlay)
                if self.view_org:
                    cv2.imshow(WINDOW_TITLE, detector.combine_overlay(frame))
                else:
                    cv2.imshow(WINDOW_TITLE, detector.combine_overlay(result))

                if detector.trigger_pause:
                    print("@@ Detector triggered pause. Space to continue.")
                    detector.trigger_pause = False
                    paused = True

                key = cv2.waitKey(FPS_MS) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = not paused
                elif key == ord('o'):
                    self.view_org = not self.view_org
                elif key == ord('s'):
                    self.skip_num = 1 + self.skip_num % 4
                elif key == ord('1'):
                    self.zoom = 1
                    cv2.resizeWindow(WINDOW_TITLE, width, height)
                elif key == ord('2'):
                    self.zoom = 2
                    cv2.resizeWindow(WINDOW_TITLE, width//2, height//2)
                elif key == ord('3'):
                    self.zoom = 3
                    cv2.resizeWindow(WINDOW_TITLE, width//3, height//3)
                elif key == ord('4'):
                    self.zoom = 4
                    cv2.resizeWindow(WINDOW_TITLE, width//4, height//4)
                end_loop_s = time.perf_counter()
                loop_s = end_loop_s - start_loop_s
        finally:
            cap.release()
            cv2.destroyAllWindows()

        print("@@ end")

if __name__ == "__main__":
    m = Main()
    m.run()

