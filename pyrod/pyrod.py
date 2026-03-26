#!/usr/bin/python
# Ensure we're running from the Virtual Env version
import os
if not "VIRTUAL_ENV" in os.environ:
    print("ERROR: Run this from venv using 'source ./venv_catd/bin/activate' first")
    exit(1)
IS_RPI = os.path.isfile("/etc/rpi-issue")

import argparse
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
IN_VIDEOS = [
    "../samples/rod1_front_randall_up_2025-03-23.mp4",
    "../samples/rod1_rear_randall_up_2025-03-23.mp4",
]

OUT_VIDEO_FILE_PATH = "outputIDX_%s.mp4" % time.strftime("%Y-%m-%d_%H-%M-%S")

GRAPH_Y_OFFSET = 10
NUM_BOTTOM_ROWS_CV_PCT = 100/720
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

    def merge(self, other_rod, weight=0.75):
        wa = weight
        wb = 1 - weight
        self.left = self.left * wa + other_rod.left * wb
        self.right = self.right * wa + other_rod.right * wb
        self.score = other_rod.score

    def center(self):
        return (self.left + self.right) / 2

    def width(self):
        return self.right - self.left

    def __repr__(self):
        return f"Rod( {self.left:.3f} -> {self.right:.3f} ; width {self.width():.3f} ; score {self.score:.3f} )"


class RodTrack:
    def __init__(self, rod, track_id):
        self.id = track_id
        self.rod = rod
        self.hits = 1
        self.misses = 0
        self.scores = [ rod.score ]
        self.active = False # Becomes true after reaching hit threshold
        self._last_temporal_score = 0

    def temporal_score(self):
        """0 for 1 hit, closer to scores' average over time."""
        temp = np.mean(self.scores) * np.log(self.hits)
        self._last_temporal_score = temp
        return temp

    def __repr__(self):
        return f"RodTrack( {self.id:3d} : hits {self.hits:3d}, miss {self.misses:3d}, temporal score {self._last_temporal_score:.3f} -> {self.rod} )"


class TemporalRodTracker:
    def __init__(self, iou_threshold=0.4, min_hits=10, max_misses=3):
        self.tracks = []
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        self.max_misses = max_misses

    def _compute_iou(self, rod1, rod2):
        # Compute IoU between two intervals
        intersection = max(0, min(rod1.right, rod2.right) - max(rod1.left, rod2.left))
        union = (rod1.right - rod1.left) + (rod2.right - rod2.left) - intersection
        return intersection / union if union > 0 else 0

    def update(self, candidates):
        matched_indices = set()

        # 1. Try to match new candidates to existing tracks
        for track in self.tracks:
            best_iou = 0
            best_cand_idx = -1

            for i, rod in enumerate(candidates):
                if i in matched_indices:
                    continue
                iou = self._compute_iou(track.rod, rod)
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_cand_idx = i

            if best_cand_idx != -1:
                # Update existing track (Simple EMA for smoothing)
                rod = candidates[best_cand_idx]
                track.rod.merge(rod)
                track.hits += 1
                track.misses = 0
                track.scores.append(rod.score)
                matched_indices.add(best_cand_idx)
            else:
                track.misses += 1

        # 2. Create new tracks for unmatched candidates
        for i, cand in enumerate(candidates):
            if i not in matched_indices:
                self.tracks.append(RodTrack(cand, self.next_id))
                self.next_id += 1

        # 3. Prune dead tracks and filter for output
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

        # Return only 'stable' tracks
        stable_tracks = [t for t in self.tracks if t.hits >= self.min_hits]
        stable_tracks.sort(
            key=lambda t: t.temporal_score(),
            reverse=True
        )
        return stable_tracks


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
        self.last_threshold = 0
        self.current_rod = None
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        self.temporal_tracker = TemporalRodTracker()


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
            rod_center = None
            score_center = cv_peaks.size / 2
            rod_left = -1
            rod_right = -1
        else:
            # score_center = self.current_rod.center()
            score_center = cv_peaks.size / 2
            rod_center = self.current_rod.center()
            rod_left = self.current_rod.left
            rod_right = self.current_rod.right

        peaks, props = scipy.signal.find_peaks(
            cv_peaks,
            prominence=0.50, # Minimum 'depth' of the valley to be considered
        )
        # print("@@ peaks", peaks, " // props", props)

        y = self.height - GRAPH_Y_OFFSET

        rod_width = self.rod_width_px
        min_width = self.rod_w_range_px[0]
        max_width = self.rod_w_range_px[1]

        left_bases = props["left_bases"]
        right_bases = props["right_bases"]
        num_peaks = len(peaks)
        best = None

        candidates = []

        for i in range(0, num_peaks):
            left_px = left_bases[i]
            right_px = right_bases[i]
            peak = peaks[i]
            peak_cv = cv_peaks[peak].item()
            # midpoint = (left_px + right_px) / 2
            width = right_px - left_px

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
                score = peak_cv * 1000 - abs(delta_center) - abs(delta_width) / 10

                # DEBUG draw
                ys = y - max(min(int(950 - score), 0), 255)
                cv2.line(self.overlay, (int(left_px), ys), (int(right_px), ys), (255, 0, 0), 3)
                text = f"{score:4.3f}"
                ys -= 10
                cv2.putText(self.overlay, text,
                    (left_px, ys),           # bottom-left coord
                    cv2.FONT_HERSHEY_DUPLEX,    # font
                    .75,                          # font scale
                    (255, 0, 0),              # color
                    1 )                         # line thickness

                # print(f"[{i}] yt {yt} > {score}")

                # # Only accept this has a suitable rod if it hasn't moved more than
                # # an acceptable margin, in this case arbitrarily the full rod size.
                # delta_center = 0
                # if rod_center is not None:
                #     delta_center = abs(peak - rod_center)
                # if delta_center <= 1 * rod_width:

                # Normalize the width around the peak
                left_px = int(peak - rod_width / 2)
                right_px = left_px + rod_width

                # if best is None:
                #     best = Rod(left_px, right_px, score)
                #     ytb = yt
                # elif score > best.score:
                #     best = Rod(left_px, right_px, score)
                #     ytb = yt

                candidates.append( Rod(left_px, right_px, score) )

        temp_best = self.temporal_tracker.update(candidates)

        # # DEBUG reprint the best match with a different color
        if temp_best:
            temp_best = temp_best[0]
            best = temp_best.rod
            print("@@ best: ", temp_best)

            text = f"{best.score:4.3f}"
            ys = y - max(min(int(950 - best.score), 0), 255)
            cv2.line(self.overlay, (int(best.left), ys), (int(best.right), ys), (0, 0, 255), 3)
            ys -= 10
            cv2.putText(self.overlay, text,
                (int(best.left), ys),           # bottom-left coord
                cv2.FONT_HERSHEY_DUPLEX,    # font
                .75,                          # font scale
                (0, 0, 255),              # color
                1 )                         # line thickness
            # print(f"best yt {ytb} > {best.score}")

        return best

    def merge_rod(self, new_rod):
        if new_rod is None:
            return
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
                left=self.weight(old.left, new_rod.left, 0.5),
                right=self.weight(old.right, new_rod.right, 0.5),
                score=self.weight(old.score, new_rod.score, 0.5)
            )
            self.current_rod = new_rod
            # print("@@ new rod:", new_rod)
            # print("@@ delta", delta_center, "<", delta_threshold, " @@ ", old, " >>> ", self.current_rod)
            # else:
            #     print("@@ delta", delta_center, ">=", delta_threshold)
        return self.current_rod

    def draw_rod(self, rod):
        if rod is None:
            return
        left_px = int(rod.left)
        right_px = int(rod.right)
        y1 = self.height - GRAPH_Y_OFFSET
        y2 = y1 - 128
        cv2.rectangle(self.overlay, (left_px, y1), (right_px, y2), (0, 255, 0), 4)

    def draw_threshold(self, threshold_y, color_threshold, dest):
        y = self.height - int(threshold_y) - GRAPH_Y_OFFSET
        cv2.line(dest, (0, y), (self.width, y), color_threshold, 1)

    def extract_roi_for_cv(self, lu, roi_q):
        # Extract the N bottom rows
        N=self.num_bottom_rows_cv
        bottom_lu = lu[-N:, :].copy()

        # Apply CLAHE to amplify local texture detail
        # We use a slightly lower clipLimit to avoid amplifying sensor noise too much
        lu_clahe = self.clahe.apply(bottom_lu)

        # Apply Histogram Stretching (Min-Max Normalization)
        # This stretches the resulting L channel to the full 0-255 range
        contrast_lu = cv2.normalize(lu_clahe, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        # zero the left and right areas we don't want to analyze
        contrast_lu[:, :roi_q] = 0
        contrast_lu[:, -roi_q:] = 0

        # for debugging, place the modified lu back into the original
        lu[-N:, :] = contrast_lu

        return bottom_lu, contrast_lu

    def is_frame_too_dark(self, roi_lu, mean_threshold=15, std_threshold=15):
        mean_intensity = np.mean(roi_lu)
        std_intensity = np.std(roi_lu)
        return mean_intensity < mean_threshold and std_intensity < std_threshold, mean_intensity, std_intensity

    def filter(self, frame):
        cv_smooth_window = 5
        epsilon = 1e-6
        roi_q = self.width // 4
        bt_cv_tuunel_threshold = 0.003

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        # lu, au, bu = cv2.split(lab)     # uint8
        lu = lab[:, :, 0]

        # Extract the N bottom rows and zero the left and right areas we don't want to analyze
        # Bottom lu is the unmodified bottom rows extracted from the source (must be pristine for
        # is_frame_too_dark to work).
        # Contrast lu is the modified CLAHE bottom rows, with the window ROI applied to it.
        # The contrasted lu is not usable in low-luminosity tunnels, too many compression artifacts.
        # Contrast_lu has an ROI with zeroed bands on the borders.
        # Even though we only filter on the middle of the image, we keep a vector of self.width
        # for ease and consistency. Our images are not very large so it's not a big penalty.
        bottom_lu, contrast_lu = self.extract_roi_for_cv(lu, roi_q)

        # We can use the original bottom CV to detect tunnels and disable rod detection.
        # The mean/std plumets in tunnels.
        is_dark, bt_mean, bt_std = self.is_frame_too_dark(bottom_lu[:, roi_q:-roi_q])
        if not is_dark:
            # Compute and smooth the CV vector
            cv_lu = self.get_cv_vectorized(contrast_lu)
            cv_lu = np.convolve(cv_lu, np.ones(cv_smooth_window)/cv_smooth_window, mode="same")

            # Adaptive thresholding
            cv_lu_inv = 1 - cv_lu
            cv_filtered = cv_lu_inv[cv_lu_inv < 1 - epsilon]
            if cv_filtered.size > 0:
                peak_threshold = np.percentile(cv_filtered, 80)
            else:
                peak_threshold = np.max(cv_lu_inv) * .95
            self.last_threshold = peak_threshold

            cv_mask = cv_lu_inv >= peak_threshold
            cv_peaks = cv_lu_inv * cv_mask
            # draw_line(self.y_np_vector(bt_cv), 0, -1, (128, 128, 128), self.overlay)
            draw_line(cv_lu_inv * 255, 0, -1, (0, 165, 255), self.overlay)
            draw_line(cv_peaks  * 255, 0, -1, (0, 255, 255), self.overlay)
            self.draw_threshold(peak_threshold * 255, (0, 255, 0), self.overlay)

            new_rod = self.find_rod_peaks(cv_peaks)
            if new_rod is not None:
                self.merge_rod(new_rod)

        # text = f"threshold {peak_threshold:4.3f}, bt_cv_median {bt_cv_median:4.3f}"
        text = f"threshold {self.last_threshold:4.3f}, bt_mean {bt_mean:4.1f}, bt_std {bt_std:4.1f}"
        cv2.putText(self.overlay, text,
            (10, 60),           # bottom-left coord
            cv2.FONT_HERSHEY_DUPLEX,    # font
            .75,                          # font scale
            (255, 255, 0),              # color
            1 )                         # line thickness

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

        parser = argparse.ArgumentParser(description="PyRod")
        parser.add_argument("-i", "--input", default="0", help="Input video")
        parser.add_argument("-o", "--output", default=OUT_VIDEO_FILE_PATH, help="Output video")
        parser.add_argument("-n", "--no-video", action="store_true", help="Skip Video Output")
        args = parser.parse_args()

        input_idx = "_"
        input_path = args.input
        if input_path.isdigit():
            input_idx = int(input_path)
            input_path = IN_VIDEOS[input_idx % len(IN_VIDEOS)]
        output_path = f"{args.output}".replace("IDX", str(input_idx))
        print("Input:", input_path)
        print("Output:", output_path, "(disabled by -n)" if args.no_video else "")

        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TITLE, 1920//2, 1080//2)
        def _mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_MOUSEMOVE:
                self.mx = x
                self.my = y
        cv2.setMouseCallback(WINDOW_TITLE, _mouse_callback)

        loop_s = 0
        init_once = True
        frame_count = 0
        paused = False

        detector = Detector2()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        cap = cv2.VideoCapture(input_path)
        writer = None
        try:
            # Get input video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps_ms = int(1000 / fps)
            self.view_org = True

            if args.no_video == False:
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
                print(f"@@ Writing {width}x{height}@{fps} fps to", output_path)

            last_frame = None
            while cap.isOpened():
                start_loop_s = time.perf_counter()
                if paused:
                    frame = last_frame.copy()
                else:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                    last_frame = frame.copy()
                    if self.view_org and frame_count == 50:
                        self.view_org = False

                _skip_num = self.skip_num
                if _skip_num > 1:
                    if frame_count % _skip_num != 0:
                        continue

                if init_once:
                    print(f"Video size: {width}x{height}")
                    detector.init_size(width, height)
                    init_once = False

                detector.init_overlay(frame)
                result = detector.filter(frame)

                #self.print_mouse_rgb(frame, detector.overlay)
                self.print_fps(loop_s, detector.overlay)

                if self.view_org:
                    show_frame = detector.combine_overlay(frame)
                else:
                    show_frame = detector.combine_overlay(result)
                cv2.imshow(WINDOW_TITLE, show_frame)

                if writer is not None and not paused:
                    writer.write(show_frame)

                if detector.trigger_pause:
                    print("@@ Detector triggered pause. Space to continue.")
                    detector.trigger_pause = False
                    paused = True

                end_loop_s = time.perf_counter()
                loop_s = end_loop_s - start_loop_s

                key = cv2.waitKey(fps_ms) & 0xFF
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
        finally:
            if writer is not None:
                writer.release()
            cap.release()
            cv2.destroyAllWindows()

        print("@@ end")

if __name__ == "__main__":
    m = Main()
    m.run()

