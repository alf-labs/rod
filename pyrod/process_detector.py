import cv2
import numpy as np
import scipy
from processor import ProcessorBase
from rod import Rod

ROD_WIDTH = 35/1280
ROI_WIDTH = 4 * ROD_WIDTH
ROI_HEIGHT = 5/12




class Detector(ProcessorBase):
    def __init__(self, locator):
        super().__init__()
        self.frame_rods = locator.frame_rods
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        self.history_mask = None

    def init_size(self, width, height):
        super().init_size(width, height)
        print(f"@@ Detector init_size")
        self.rod_width_px = int(ROD_WIDTH * width)
        self.roi_width = int(ROI_WIDTH * width)
        self.roi_height = int(ROI_HEIGHT * height)
        print(f"ROI Detector: {self.roi_width}x{self.roi_height}")

    def init_overlay(self, frame):
        super().init_overlay(frame)

    def extract_roi(self, lu, rod_x_ctr):
        roi_x_left  = int(rod_x_ctr - self.roi_width / 2)
        roi_x_right = roi_x_left + self.roi_width

        roi_lu = lu[-self.roi_height:, roi_x_left:roi_x_right].copy()

        # Apply CLAHE to amplify local texture detail
        # We use a slightly lower clipLimit to avoid amplifying sensor noise too much
        lu_clahe = self.clahe.apply(roi_lu)

        # Apply Histogram Stretching (Min-Max Normalization)
        # This stretches the resulting L channel to the full 0-255 range
        contrast_lu = cv2.normalize(lu_clahe, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        # for debugging, place the modified lu back into the original
        lu[-self.roi_height:, roi_x_left:roi_x_right] = contrast_lu

        return roi_x_left, roi_lu, contrast_lu

    def draw_roi_bounds(self, roi_x_left, rod_x_ctr):
        cv2.rectangle(self.overlay,
            (roi_x_left, self.height - self.roi_height),
            (roi_x_left + self.roi_width, self.height - 1),
            color=(255, 255, 0),
            thickness=1)

        cv2.circle(self.overlay,
            (rod_x_ctr, self.height - 2),
            radius=5,
            color=(0, 255, 0),
            thickness=-1)

    def track_rod_upward_unused(self, roi_lu, start_x, start_y, target_width, search_margin):
        """
        Tracks a bending rod from bottom to top using local CV valleys.
        """
        search_margin = int(search_margin)
        height, width = roi_lu.shape
        if start_y < 0:
            start_y += height
        current_center = start_x
        path = []

        # We move from start_y (bottom) up to 0 (top)
        for y in range(start_y, -1, -1):
            # 1. Define a local Search Window to handle the bend
            # Only look near the previous row's center
            win_left = max(0, int(current_center - target_width - search_margin))
            win_right = min(width, int(current_center + target_width + search_margin))

            row_slice = roi_lu[y, win_left:win_right]
            # Convert from 0..255 uint8 to 0..1 float range.
            row_slice = row_slice.astype(np.float32) / 255.0

            # 2. Compute local Coefficient of Variation
            # We use a small sliding window (e.g., 5px) to compute local std/mean
            # This highlights 'texture' vs 'smoothness'
            kernel_size = 5
            krn = np.ones(kernel_size)/kernel_size
            mean = np.convolve(row_slice, krn, mode='same')
            conv = np.convolve(row_slice**2, krn, mode='same') - mean**2
            std = np.sqrt(conv)
            cv = std / (mean + 1e-6)

            # 3. Find the 'Valley' in this row's CV signal
            # Use find_peaks on inverted CV
            peaks, props = scipy.signal.find_peaks(-cv, prominence=0.01, width=(target_width*0.5, target_width*1.5))

            if len(peaks) > 0:
                # Find the peak closest to our expected center
                best_peak_idx = peaks[np.argmin(np.abs(peaks + win_left - current_center))]

                # Convert local window index back to global image index
                actual_center = best_peak_idx + win_left

                # Update center for the next row (adds the 'tracking' memory)
                current_center = actual_center

                path.append({
                    'y': y,
                    'x': int(actual_center),
                    'left': float(win_left + props['left_ips'][0]),
                    'right': float(win_left + props['right_ips'][0])
                })
            else:
                # 4. Stopping Condition: If no valley is found for 3 rows, we hit the top
                if y < start_y - 10: # Allow a small buffer at the start
                    break

        print("@@ path", path)
        return path

    def draw_path_unused(self, path, roi_x_left):
        y_offset = self.height - self.roi_height
        for p in path:
            x = int(p["x"] + roi_x_left)
            y = int(p["y"] + y_offset)
            left = int(p["left"])
            right = int(p["right"])

            cv2.line(self.overlay, (left, y), (right, y), (255, 0, 0), 1)
            cv2.circle(self.overlay,
                (x, y),
                radius=3,
                color=(0, 255, 0),
                thickness=-1)

    def find_rod_boundaries_floodfill_unused(self, luminance, tracked_x, tracked_y, seed_threshold=20, connectivity=4):
        """
        Find the rod's boundaries using flood fill.

        Args:
            luminance (np.ndarray): Grayscale Luminance channel (H, W).
            tracked_x (int): X-coordinate of the tracked position (bottom-middle).
            tracked_y (int): Y-coordinate of the tracked position (bottom-middle).
            seed_threshold (int): Max luminance difference allowed for region growing.
            connectivity (int): 4 or 8 for pixel connectivity.

        Returns:
            dict: {"top": top_y, "bottom": bottom_y, "left": left_x, "right": right_x}
        """
        # Convert to 3-channel for floodFill (OpenCV requires 3 channels)
        luminance_3ch = cv2.cvtColor(luminance, cv2.COLOR_GRAY2BGR)

        # Create a mask (must be 2 pixels larger in width and height)
        h, w = luminance.shape
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

        # # For DEBUG, try a basic binary mask
        # trigger_luminance = np.mean(luminance)
        # trigger_luminance = (trigger_luminance + luminance[tracked_y, tracked_x]) // 2
        # filled_mask = luminance > trigger_luminance
        tracked_lum = luminance[tracked_y, tracked_x]
        trigger_lum = int(np.mean(luminance) + tracked_lum) // 2
        seed_threshold = int(abs(tracked_lum - trigger_lum)) + 5
        print(f"@@ tracked_lum {tracked_lum} - trigger_lum {trigger_lum} = seed_threshold {seed_threshold}")

        # Seed point (note: floodFill uses (x, y) format)
        seed = (tracked_x, tracked_y)

        # Fill the region
        cv2.floodFill(
            luminance_3ch,  # Input image (3-channel)
            mask,           # Output mask
            seed,           # Seed point (x, y)
            (255, 255, 255),      # New value (black, but we only care about the mask)
            # loDiff=(seed_threshold, seed_threshold, seed_threshold),  # Lower bound for similarity
            upDiff=(seed_threshold, seed_threshold, seed_threshold),  # Upper bound for similarity
            flags=connectivity  # 4 or 8 connectivity
        )

        # Extract the filled region from the mask (remove the 1-pixel border)
        filled_mask = mask[1:-1, 1:-1]

        # Find the bounding box of the filled region
        rows = np.any(filled_mask, axis=1)
        cols = np.any(filled_mask, axis=0)
        top_y = np.argmax(rows)
        bottom_y = h - 1 - np.argmax(np.flip(rows))
        left_x = np.argmax(cols)
        right_x = w - 1 - np.argmax(np.flip(cols))

        return {
            "top": top_y,
            "bottom": bottom_y,
            "left": left_x,
            "right": right_x,
        }, filled_mask

    def find_rod_by_threshold(self, roi_lu, tracked_x, tracked_y):
        # Try a basic binary mask
        median_luminance = np.median(roi_lu)
        trigger_luminance = (median_luminance + roi_lu[tracked_y, tracked_x]) // 2
        mask = roi_lu > trigger_luminance
        mask_f32 = mask.astype(np.float32)

        # Erode and dilate
        # Kernel choices: 3x3 typical, or 5x1 (vertical band) to favor vertical features.
        kernel = np.ones((5, 1), np.uint8)
        mask_f32 = cv2.morphologyEx(mask_f32, cv2.MORPH_OPEN, kernel, iterations=1)

        return mask_f32

    def temporal_smooth_mask(self, mask_f32, history_mask, weight_new):
        if history_mask is None:
            new_history = mask_f32
        else:
            # Weighted average: history + new evidence
            new_history = cv2.addWeighted(mask_f32, weight_new, history_mask, 1.0 - weight_new, 0)

        # To get a binary mask back, we threshold the 'probability'
        # Only pixels that have been 'white' consistently stay above 0.5
        mask_u8 = (new_history > 0.75).astype(np.uint8) * 255

        return new_history, mask_u8

    def keep_contiguous_rod(self, mask_u8, seed_x, seed_y):
        h, w = mask_u8.shape
        filled_mask = np.zeros((h, w), dtype=np.uint8)

        # # 2. Safety check: Is the tracker point actually on a white pixel?
        # # If the tracker is on a black pixel (due to a tiny gap), floodFill won't start.
        # if mask[seed_y, seed_x] == 0:
        #     # Optional: Search in a tiny 3x3 neighborhood for the nearest white pixel
        #     return mask # Or handle as a "lost tracking" state

        # floodfill modifies the source; that's fine as we're not keeping it anyway
        # -- not needed here -- temp_mask = mask.copy()
        # floodFill(image, mask, seedPoint, newVal)
        # Note: newVal=255 means we turn the rod white on our temp_mask
        # cv2.floodFill(mask_u8, None, (seed_x, seed_y), 255)

        # 4. The 'Magic Trick': Bitwise Comparison
        # After floodFill, the connected rod is still 255, but we need to isolate
        # what changed. A simpler way: use the 'mask' parameter of floodFill
        # or just use the following logic:

        # Create an empty mask for the result
        just_the_rod = np.zeros((h + 2, w + 2), np.uint8) # floodFill needs +2 size
        cv2.floodFill(mask_u8, just_the_rod, (seed_x, seed_y), 255)

        # The 'just_the_rod' mask now contains the filled area (at indices 1:-1)
        return just_the_rod[1:-1, 1:-1] * 255

    def draw_mask(self, mask, color, roi_x_left):
        height, width = mask.shape
        overlay_view = self.overlay[-height:, roi_x_left:roi_x_left + width]
        overlay_view[mask != 0] = color

    def filter(self, frame_index, frame):
        if frame_index >= 0 and frame_index < len(self.frame_rods):
            rod = self.frame_rods[frame_index]
        else:
            return frame

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lu = lab[:, :, 0]

        rod_x_ctr = int(rod.center())
        roi_x_left, roi_lu, contrast_lu = self.extract_roi(lu, rod_x_ctr)
        self.draw_roi_bounds(roi_x_left, rod_x_ctr)

        # path = self.track_rod_upward(roi_lu, rod_x_ctr - roi_x_left, -1,
        #     target_width=self.rod_width_px,
        #     search_margin=self.rod_width_px / 2)
        # self.draw_path(path, roi_x_left)

        # result, filled_mask = self.find_rod_boundaries_floodfill(roi_lu,
        #     rod_x_ctr - roi_x_left,
        #     self.roi_height - 1)
        # self.draw_mask(filled_mask, (0, 0, 255), roi_x_left)
        # print("@@ ", frame_index, result)

        mask_f32 = self.find_rod_by_threshold(roi_lu,
            rod_x_ctr - roi_x_left,
            self.roi_height - 1)
        self.history_mask, mask_u8 = self.temporal_smooth_mask(
            mask_f32, self.history_mask, weight_new=0.25)
        mask_u8 = self.keep_contiguous_rod(mask_u8,
            rod_x_ctr - roi_x_left,
            self.roi_height - 1)
        self.draw_mask(mask_u8, (0, 0, 255), roi_x_left)

        return cv2.cvtColor(lu, cv2.COLOR_GRAY2BGR)

    def export(self, filename):
        super().export(filename)

    def release(self):
        super().release()

