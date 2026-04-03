import cv2
import math
import numpy as np
import scipy
from processor import ProcessorBase
from rod import Rod
from rod_tracker import TemporalRodTracker

GRAPH_Y_OFFSET = 10
NUM_BOTTOM_ROWS_CV_PCT = 100/720
ROD_WIDTH = 35/1280
ROD_W_RANGE = (20/1280, 60/1280)

ROI_WIDTH_PCT = 0.3

TRACKER_IOU_PCT=0.25
TRACKER_MIN_HITS=7
TRACKER_MAX_MISS=3

class LocatorBase(ProcessorBase):
    def __init__(self):
        super().__init__()
        self.frame_rods = []
        self.frame_tunnel_metric = {}
        self.downscale = 1  # can be either 1 or 2

    def init_size(self, width, height):
        super().init_size(width, height)
        # Width, as a fraction of the screen width
        self.rod_width_px = int(ROD_WIDTH * width)
        self.rod_w_range_px = ( int(ROD_W_RANGE[0] * width), int(ROD_W_RANGE[1] * width) )
        print("Rod Width PX: ", self.rod_width_px, "in range", self.rod_w_range_px)


    def init_overlay(self, frame):
        super().init_overlay(frame)

    def append_frame_rod(self, new_rod):
        last_rod = None
        last_idx = 0
        if self.frame_rods:
            last_rod = self.frame_rods[-1]
            last_idx = last_rod.frame
        new_idx = new_rod.frame

        if new_idx > last_idx + 1:
            if last_rod is None:
                # Gap before the first rod. We just fill as-is.
                for idx in range(last_idx + 1, new_idx):
                    rod = new_rod.dupAtFrame(idx)
                    if idx in self.frame_tunnel_metric:
                        rod.tunnel_metric = self.frame_tunnel_metric[idx]
                    self.frame_rods.append(rod)
            else:
                # Otherwise we have a frame gap, which we want to close by interpolation.
                for idx in range(last_idx + 1, new_idx):
                    rod = last_rod.dupInterpolateTo(idx, new_rod)
                    if idx in self.frame_tunnel_metric:
                        rod.tunnel_metric = self.frame_tunnel_metric[idx]
                    self.frame_rods.append(rod)

        # Append rod for frame new_idx
        if new_idx in self.frame_tunnel_metric:
            new_rod.tunnel_metric = self.frame_tunnel_metric[new_idx]
        self.frame_rods.append(new_rod)

    def draw_rod(self, rod, new_rod=True):
        if rod is None:
            return
        left_px = int(rod.left)
        right_px = int(rod.right)
        y1 = self.height - GRAPH_Y_OFFSET
        y2 = y1 - 128
        color = (0, 255, 0) if new_rod else (0, 0, 255)
        cv2.rectangle(self.overlay, (left_px, y1), (right_px, y2), color, 4)

    def filter(self, frame_index, frame):
        return super().filter(frame_index, frame)

    def export(self):
        return super().export()

    def release(self):
        super().release()


class LocatorGen(LocatorBase):
    def __init__(self):
        super().__init__()
        self.last_cv_lu = None
        self.last_threshold = 0
        self.current_rod = None
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        cv_smooth_window = 5
        self.cv_smooth_kernel = np.ones(cv_smooth_window)/cv_smooth_window
        self.temporal_tracker = TemporalRodTracker(
            iou_threshold=TRACKER_IOU_PCT,
            min_hits=TRACKER_MIN_HITS,
            max_misses=TRACKER_MAX_MISS
        )
        self.frame_rods = []

    def init_size(self, width, height):
        if width > 1280 and width % 2 == 0 and height % 2 == 0:
            self.downscale = 2
            width = width // 2
            height = height // 2
            print(f"Locator downscaling by factor {self.downscale} to {width}x{height}")

        super().init_size(width, height)
        print(f"@@ Locator init_size")

        # Number of rows to scan at the bottom to get the vertical per-band CV
        self.num_bottom_rows_cv = int(NUM_BOTTOM_ROWS_CV_PCT * height)

        self.roi_width_px = int(ROI_WIDTH_PCT * width)
        self.roi_q = int((width - self.roi_width_px) // 2)
        print("ROI Width PX: ", self.roi_width_px, "px with side bands", self.roi_q, "px")

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

    def fit_rod_bounds(self, left_px, right_px, peak, rod_width):
        left_px = int(left_px)
        right_px = int(right_px)
        new_left = int(peak - rod_width / 2)
        if new_left < left_px:
            new_left = left_px
        new_right = new_left + rod_width
        if new_right > right_px:
            new_right = right_px
            new_left = new_right - rod_width
        return new_left, new_right

    def find_rod_peaks(self, cv_peaks):
        if self.current_rod is None:
            rod_center = None
            score_center = cv_peaks.size / 2
            rod_left = -1
            rod_right = -1
        else:
            score_center = cv_peaks.size / 2
            rod_center = self.current_rod.center()
            rod_left = self.current_rod.left
            rod_right = self.current_rod.right

        peaks, props = scipy.signal.find_peaks(
            cv_peaks,
            prominence=0.50, # Minimum 'depth' of the valley to be considered
        )

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
            width = right_px - left_px

            # Apply Width Constraints
            cond_width = min_width <= width <= max_width
            if cond_width:

                # Calculate Center Score.
                # Score starts with the peak's CV value, so we want the highest one.
                # It is degraded (lowered) by the distance from the center,
                # and further degraded by the absolute difference from the ideal width.
                delta_center = peak - score_center
                delta_width = width - rod_width
                score = peak_cv * 1000 - abs(delta_center) * 2 - abs(delta_width)

                # If the width is much wider than the target, try to
                # normalize the width around the peak whilst keeping
                # in the current left/right boundaries
                if cond_width > rod_width:
                    left_px, right_px = self.fit_rod_bounds(left_px, right_px, cond_width, rod_width)

                candidates.append( Rod(left_px, right_px, score) )

        temp_best = self.temporal_tracker.update(candidates)
        best_id = -1
        if temp_best:
            temp_best = temp_best[0]
            best_id = temp_best.id
            best = temp_best.rod
            # print(f"@@ Best: {temp_best}")

        if self.compute_overlay:
            y = self.height - GRAPH_Y_OFFSET
            ys = y - 255
            for t in self.temporal_tracker.tracks:
                color = (0, 0, 255) if t.id == best_id else (255, 255, 0)

                iou = 0
                if self.current_rod is not None:
                    iou = t.rod.iou(self.current_rod)

                # ys = y - min(max(int(950 - t.rod.score), 0), 255)
                ys -= 5
                text1 = f"{iou:4.2f}"
                text2 = f"{t.rod.score:4.3f}"
                cv2.line(self.overlay, (int(t.rod.left), ys), (int(t.rod.right), ys), color, 3)
                ys -= 10
                cv2.putText(self.overlay, text2,
                    (int(t.rod.left), ys),      # bottom-left coord
                    cv2.FONT_HERSHEY_DUPLEX,    # font
                    .75,                        # font scale
                    color,                      # color
                    1 )                         # line thickness
                cv2.putText(self.overlay, text1,
                    (int(t.rod.left), ys - 30), # bottom-left coord
                    cv2.FONT_HERSHEY_DUPLEX,    # font
                    .75,                        # font scale
                    color,                      # color
                    1 )                         # line thickness

        return best

    def draw_threshold(self, threshold_y, color_threshold, dest):
        y = self.height - int(threshold_y) - GRAPH_Y_OFFSET
        cv2.line(dest, (0, y), (self.width, y), color_threshold, 1)

    def draw_line(self, source, channel_index, y, color, dest):
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

        # mean_norm = min(mean_intensity / mean_threshold, 1.0)
        # std_norm = min(std_intensity / std_threshold, 1.0)
        mean_norm = mean_intensity / mean_threshold
        std_norm = std_intensity / std_threshold
        tunnel_metric = min(mean_norm, std_norm)
        return tunnel_metric < 1.0, tunnel_metric

    def filter(self, frame_index, frame):
        epsilon = 1e-6
        roi_q = self.roi_q
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
        is_dark, tunnel_metric = self.is_frame_too_dark(bottom_lu[:, roi_q:-roi_q])
        self.frame_tunnel_metric[frame_index] = tunnel_metric
        new_rod = None
        if not is_dark:
            # Compute and smooth the CV vector
            cv_lu = self.get_cv_vectorized(contrast_lu)
            cv_lu = np.convolve(cv_lu, self.cv_smooth_kernel, mode="same")

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
            if self.compute_overlay:
                # self.draw_line(self.y_np_vector(bt_cv), 0, -1, (128, 128, 128), self.overlay)
                self.draw_line(cv_lu_inv * 255, 0, -1, (0, 165, 255), self.overlay)
                self.draw_line(cv_peaks  * 255, 0, -1, (0, 255, 255), self.overlay)
                self.draw_threshold(peak_threshold * 255, (0, 255, 0), self.overlay)

            new_rod = self.find_rod_peaks(cv_peaks)
            if new_rod is not None:
                self.current_rod = new_rod
                self.append_frame_rod( new_rod.dupAtFrame(frame_index, tunnel_metric) )

        if self.compute_overlay:
            # text = f"threshold {peak_threshold:4.3f}, bt_cv_median {bt_cv_median:4.3f}"
            text = f"threshold: {self.last_threshold:4.3f}, tunnel: {tunnel_metric:4.3f}"
            cv2.putText(self.overlay, text,
                (10, 60),           # bottom-left coord
                cv2.FONT_HERSHEY_DUPLEX,    # font
                .75,                          # font scale
                (255, 255, 0),              # color
                1 )                         # line thickness
            self.draw_rod(self.current_rod, new_rod is not None)

        return cv2.cvtColor(lu, cv2.COLOR_GRAY2BGR)

    def pre_release(self):
        super().pre_release()
        for rod in self.frame_rods:
            rod.apply_scale(self.downscale)

    def export(self):
        print(f"@@ Locator export")
        content = [ rod.toJson() for rod in self.frame_rods ]
        return { "locator": content }

    def release(self):
        super().release()


class LocatorRdr(LocatorBase):
    def __init__(self):
        super().__init__()

    def init_size(self, width, height):
        super().init_size(width, height)

    def init_overlay(self, frame):
        super().init_overlay(frame)

    def read_json(self, exported_content):
        print(f"@@ LocatorRdr JSON load")
        # exported_content should contain one array of Rod.toJson().
        for entry in exported_content:
            self.append_frame_rod(Rod.fromJson(entry))
        print(f"@@ LocatorRdr loaded {len(self.frame_rods)} entries from JSON")
        self.next_processor_requested = True

    def filter(self, frame_index, frame):
        if frame_index >= 0 and frame_index < len(self.frame_rods):
            rod = self.frame_rods[frame_index]
            if self.compute_overlay:
                self.draw_rod(rod)
        return frame

    def pre_release(self):
        print(f"@@ LocatorRdr pre-release")
        self.fix_sudden_short_movements()

    def fix_sudden_short_movements(self):
        nf = len(self.frame_rods)
        lc = self.frame_rods[0].center()
        for i1 in range(0, nf):
            f1 = self.frame_rods[i1]
            c1 = f1.center()
            d1 = c1 - lc
            if abs(d1) > 2 * self.rod_width_px:
                # We have a variation in rod position that is significant.
                # If we can find a signification variation in the opposite directiom
                # within 1 second of frames, we cancel the variation using interpolation.
                s2 = -1 * np.sign(d1)
                l2 = c1
                for i2 in range(i1 + 1, i1 + 30):
                    f2 = self.frame_rods[i2]
                    c2 = f2.center()
                    d2 = c2 - l2
                    l2 = c2
                    if abs(d2) > 2* self.rod_width_px and np.sign(d2) == s2:
                        # This is the significant variation in the opposite direction.
                        # 'lc' is the starting center we want to keep (not 'c1')
                        # and 'c2' is the end center we want to keep.
                        # All rods centers from i1 (included) up to i2-1 need to be changed.
                        print(f"Fix center for frames {i1}...{i2-1}")
                        dc = c2 - lc
                        ni = i2 - i1
                        si = i1
                        for i in range(0, ni):
                            c = lc + dc * float(i) / ni
                            self.frame_rods[i1 + i].recenter(c)
            # print(f"@@ [{i1:05d}] --> {c1:5.2f} /\ {d1:5.2f} { '*' * (round(math.log(abs(d1) + 1e-6)))}")
            lc = c1

        c = np.array([fr.center() for fr in self.frame_rods])
        dc = np.diff(c)
        print(f"@@ dc min {np.min(dc)},  mean {np.mean(dc)},  median {np.median(dc)},  max {np.max(dc)},")
        counts, bin_edges = np.histogram(dc, bins=10)
        print(f"@@ dc counts {counts}")
        print(f"@@ bin_edges {bin_edges}")


    def export(self):
        return super().export()

    def release(self):
        super().release()

