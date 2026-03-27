import cv2
import numpy as np
import scipy
import skimage
from skimage.restoration import inpaint
from processor import ProcessorBase
from rod import Rod

ROD_WIDTH = 35/1280
ROI_WIDTH = 8 * ROD_WIDTH
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

    def init_overlay(self, frame, view_mask):
        super().init_overlay(frame, view_mask)

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
        flood_mask = np.zeros((h + 2, w + 2), np.uint8) # floodFill needs +2 size
        cv2.floodFill(mask_u8, flood_mask, (seed_x, seed_y), 255)

        return flood_mask[1:-1, 1:-1] * 255

    def draw_mask(self, mask, color, roi_x_left):
        height, width = mask.shape
        overlay_view = self.overlay[-height:, roi_x_left:roi_x_left + width]
        overlay_view[mask != 0] = color

    def inpaint_rod_biharmonic_unused(self, rgb_image, mask_u8):
        # Convert mask to boolean (True where rod is)
        mask_b = mask_u8.astype(bool)

        # Convert RGB image to float in [0, 1]
        rgb_f32 = rgb_image.astype(np.float32) / 255.0

        # Apply biharmonic inpainting
        inpainted_f32 = inpaint.inpaint_biharmonic(rgb_f32, mask_b, channel_axis=-1)

        # Convert back to uint8
        inpainted_u8 = (inpainted_f32 * 255).astype(np.uint8)
        return inpainted_u8

    def measure_rod_width(self, mask_u8, tracked_x, tracked_y):
        row = mask_u8[tracked_y, :]  # Extract the row
        index255 = np.where(row == 255)[0]  # Find all rod pixels in the row
        # index0 = np.where(row == 0)[0]  # Find all rod pixels in the row
        # print(f"@@ idx0: {index0} // idx255: {index255}")
        left255 = index255[0]
        right255 = index255[-1]
        # print(f"@@ {left255} -> {right255} // idx255: {index255}")

        threshold = 1
        left0 = np.argmax(row[:tracked_x] > threshold)
        left0 = min(left0, left255)
        right0 = tracked_x + np.argmax(row[tracked_x:] <= threshold)
        right0 = max(right255, right0)

        print(f"@@ {left0} >> {left255} == {right255} [{right255 - left255}] >> {right0} [{right0 - left0}]")

        # print(f"@@ left0 {left0} // {row[:tracked_x] > threshold}")
        # left = max(0, tracker_x - len(row[:tracker_x]) + left)

        # # Find right boundary: first index where row <= threshold (from center to right)
        # right = tracker_x + np.argmax(row[tracker_x:] <= threshold)
        # right = min(len(row) - 1, right)

        # return right - left + 1  # Width

    def filter(self, frame_index, frame):
        if frame_index >= 0 and frame_index < len(self.frame_rods):
            rod = self.frame_rods[frame_index]
        else:
            return frame

        if rod.isTunnel():
            return frame

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lu = lab[:, :, 0]

        rod_x_ctr = int(rod.center())
        roi_x_left, roi_lu, contrast_lu = self.extract_roi(lu, rod_x_ctr)
        self.draw_roi_bounds(roi_x_left, rod_x_ctr)

        mask_f32 = self.find_rod_by_threshold(roi_lu,
            rod_x_ctr - roi_x_left,
            self.roi_height - 1)
        self.history_mask, mask_u8 = self.temporal_smooth_mask(
            mask_f32, self.history_mask, weight_new=0.25)
        mask_u8 = self.keep_contiguous_rod(mask_u8,
            rod_x_ctr - roi_x_left,
            self.roi_height - 1)

        roi_x_right = roi_x_left + self.roi_width
        roi_rgb = frame[-self.roi_height:, roi_x_left:roi_x_right]

        h_dilate_width = 9
        kernel_h = np.ones((1, h_dilate_width), np.uint8)
        mask_u8 = cv2.dilate(mask_u8, kernel_h, iterations=1)
        h_blur_width = 15
        mask_u8 = cv2.GaussianBlur(mask_u8, (h_blur_width, 1), 0)

        if self.view_mask:
            self.draw_mask(mask_u8, (0, 0, 255), roi_x_left)

        self.measure_rod_width(mask_u8,
            rod_x_ctr - roi_x_left,
            self.roi_height - 1)

        # inpainted = cv2.inpaint(roi_rgb, mask_u8, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        # inpainted = cv2.inpaint(roi_rgb, mask_u8, inpaintRadius=5, flags=cv2.INPAINT_NS)
        # inpainted = self.inpaint_rod_biharmonic(roi_rgb, mask_u8)
        # frame[-self.roi_height:, roi_x_left:roi_x_right] = inpainted

        if self.view_mask:
            return cv2.cvtColor(lu, cv2.COLOR_GRAY2BGR)
        else:
            return frame

    def export(self, filename):
        super().export(filename)

    def release(self):
        super().release()

