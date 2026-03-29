import cv2
import numpy as np
import scipy
import skimage
from skimage.restoration import inpaint
from processor import ProcessorBase
from rod import Rod

ROD_WIDTH = 35/1280
ROI_WIDTH = 4 * ROD_WIDTH
ROI_HEIGHT = 5/12
SKEW_PCT = 45/180



class Detector(ProcessorBase):
    def __init__(self, locator, inpainting=True):
        super().__init__()
        self.frame_rods = locator.frame_rods
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        self.do_inpainting = inpainting
        self.history_mask = None
        self.view_mask = not inpainting

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

    def find_rod_by_threshold(self, roi_lu, tracked_x, tracked_y):
        # Try a basic binary mask
        median_luminance = np.median(roi_lu)
        trigger_luminance = (median_luminance + roi_lu[tracked_y, tracked_x]) // 2
        mask_b = roi_lu > trigger_luminance
        # mask_f32 = mask_u8.astype(np.float32)
        mask_u8 = mask_b.astype(np.uint8) * 255

        # Erode and dilate
        # Kernel choices: 3x3 typical, or 5x1 (vertical band) to favor vertical features.
        kernel = np.ones((5, 3), np.uint8)
        # mask_f32 = cv2.morphologyEx(mask_f32, cv2.MORPH_OPEN, kernel, iterations=1)
        # return mask_f32
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)

        # Erode the mask horizontally a bit more
        h_dilate_width = 15
        kernel_h = np.ones((1, h_dilate_width), np.uint8)
        mask_u8 = cv2.dilate(mask_u8, kernel_h, iterations=1)

        return mask_u8

    def temporal_smooth_mask_f32(self, mask_f32, history_mask, weight_new):
        if history_mask is None:
            new_history = mask_f32
        else:
            # Weighted average: history + new evidence
            new_history = cv2.addWeighted(mask_f32, weight_new, history_mask, 1.0 - weight_new, 0)

        # To get a binary mask back, we threshold the 'probability'
        # Only pixels that have been 'white' consistently stay above 0.5
        mask_u8 = (new_history > 0.75).astype(np.uint8) * 255

        return new_history, mask_u8

    def temporal_smooth_mask_u8(self, mask_u8, history_u8, weight_u8):
        if history_u8 is None:
            return mask_u8, mask_u8
        inv_weight_u8 = 256 - weight_u8

        # Formula: (History * (256 - alpha) + New * alpha) / 256
        acc = (history_u8.astype(np.uint16) * inv_weight_u8) + (mask_u8.astype(np.uint16) * weight_u8)
        # Bit-shift to divide by 256
        new_history = (acc >> 8).astype(np.uint8)

        # Original 0.75 threshold becomes 191 (0.75 * 255)
        # cv2.threshold is much faster than numpy boolean comparisons
        _, mask_u8 = cv2.threshold(new_history, 191, 255, cv2.THRESH_BINARY)

        return new_history, mask_u8

    def keep_contiguous_rod(self, mask_u8, seed_x, seed_y):
        h, w = mask_u8.shape
        flood_mask = np.zeros((h + 2, w + 2), np.uint8) # floodFill needs +2 size

         # Note: use FLOODFILL_MASK_ONLY to avoid modifying mask_f32 (we don't nede it)
        # thus this ignores newVal. Just kept it for the sake of example.
        cv2.floodFill(
            mask_u8,
            flood_mask,
            (seed_x, seed_y),
            newVal=255,
            flags=cv2.FLOODFILL_MASK_ONLY)

        return flood_mask[1:-1, 1:-1] * 255

    def draw_mask(self, mask, roi_x_left):
        height, width = mask.shape
        overlay_view = self.overlay[-height:, roi_x_left:roi_x_left + width]
        overlay_view[mask >    1] = (0, 255, 255)
        overlay_view[mask >  128] = (0, 128, 255)
        overlay_view[mask == 255] = (0,   0, 255)

    def inpaint_rod_biharmonic_unused(self, rgb_image, mask_u8):
        # Convert mask to boolean (True where rod is)
        mask_b = mask_u8 > 16

        # Convert RGB image to float in [0, 1]
        rgb_f32 = rgb_image.astype(np.float32) / 255.0

        # Apply biharmonic inpainting
        inpainted_f32 = inpaint.inpaint_biharmonic(rgb_f32, mask_b,
            split_into_regions=True,
            channel_axis=-1)

        # Convert back to uint8
        inpainted_u8 = (inpainted_f32 * 255).astype(np.uint8)
        return inpainted_u8

    def measure_rod_width(self, mask_u8, tracked_x, tracked_y):
        row = mask_u8[tracked_y, :]  # Extract the row
        index255 = np.where(row == 255)[0]  # Find all rod pixels in the row
        if len(index255) == 0:
            return 0, 0

        left255 = index255[0]
        right255 = index255[-1]
        # print(f"@@ {left255} -> {right255} // idx255: {index255}")
        # return left255, right255

        threshold = 1
        left0 = np.argmax(row[:tracked_x] > threshold)
        left0 = min(left0, left255 - self.rod_width_px)
        right0 = tracked_x + np.argmax(row[tracked_x:] <= threshold)
        right0 = max(right255 + self.rod_width_px, right0)

        # print(f"@@ {left0} >> {left255} == {right255} [{right255 - left255}] >> {right0} [{right0 - left0}]")
        # TBD we could detect when the width becomes suddenly much larger and contain it
        # using an historical center tracker?

        return left0, right0

    def extract_deskewed_left(self, wide_roi_rgb, skew_px):
        """
        Extracts a skewed parallelogram and straightens it.
        skew_px: how many pixels the bottom moves relative to the top
                (must be positive = bottom is further right).
        """
        h, w, _ = wide_roi_rgb.shape
        dest_w = w - skew_px

        # 1. Define the 3 source points (The Parallelogram)
        # [Top-Left, Top-Right, Bottom-Left]
        src_pts = np.array([
            [0, 0],
            [dest_w, 0],
            [skew_px, h]
        ], dtype=np.float32)

        # 2. Define the 3 destination points (The Straight Rectangle)
        dst_pts = np.array([
            [0, 0],
            [dest_w, 0],
            [0, h]
        ], dtype=np.float32)

        # 3. Calculate the Transformation Matrix
        mat = cv2.getAffineTransform(src_pts, dst_pts)

        # 4. Warp the image to a new buffer
        straight_strip = cv2.warpAffine(
            src=wide_roi_rgb,
            M=mat,
            dsize=(dest_w, h),
            flags=cv2.INTER_CUBIC)

        # 5. Pad on the left to retain the original size.
        # pad constant mode = can add zeroes
        # pad edge mode = dup 1st column
        pad_width = ((0, 0), (skew_px, 0), (0, 0))
        return np.pad(straight_strip, pad_width, mode='edge')

    def extract_deskewed_right(self, wide_roi_rgb, skew_px):
        """
        Extracts a skewed parallelogram and straightens it.
        skew_px: how many pixels the bottom moves relative to the top
                (must be positive = bottom is further left).
        """
        h, w, _ = wide_roi_rgb.shape
        dest_w = w - skew_px

        # 1. Define the 3 source points (The Parallelogram)
        # [Top-Left, Top-Right, Bottom-Left]
        src_pts = np.array([
            [skew_px, 0],
            [w, 0],
            [0, h]
        ], dtype=np.float32)

        # 2. Define the 3 destination points (The Straight Rectangle)
        dst_pts = np.array([
            [0, 0],
            [dest_w, 0],
            [0, h]
        ], dtype=np.float32)

        # 3. Calculate the Transformation Matrix
        mat = cv2.getAffineTransform(src_pts, dst_pts)

        # 4. Warp the image to a new buffer
        straight_strip = cv2.warpAffine(
            src=wide_roi_rgb,
            M=mat,
            dsize=(dest_w, h),
            flags=cv2.INTER_CUBIC)

        # 5. Pad on the right to retain the original size.
        # pad constant mode = can add zeroes
        # pad edge mode = dup 1st column
        pad_width = ((0, 0), (0, skew_px), (0, 0))
        return np.pad(straight_strip, pad_width, mode='edge')

    def inpaint_dual_mirror(self, wide_roi_rgb, wide_mask_u8, left0, right0):
        h, w, _ = wide_roi_rgb.shape

        # 1. Normalize Mask to a 0.0-1.0 float factor
        # Expand dims to (H, W, 1) so it multiplies across R, G, and B channels
        alpha = wide_mask_u8.astype(np.float32) / 255.0
        alpha = np.expand_dims(alpha, axis=-1)

        # 2. Create the Mirrored X-Coordinate Maps
        # x is [0, 1, 2, ..., W-1]
        x = np.arange(w)

        # Scalar version (Rod is perfectly vertical)
        idx_l = np.clip(2 * left0 - x, 0, w - 1).astype(np.int32)
        idx_r = np.clip(2 * right0 - x, 0, w - 1).astype(np.int32)

        # Sample mirrored images
        img_l = wide_roi_rgb[:, idx_l, :]
        img_r = wide_roi_rgb[:, idx_r, :]

        skew_px = int(SKEW_PCT * h)
        img_l = self.extract_deskewed_left(img_l, skew_px)
        img_r = self.extract_deskewed_right(img_r, skew_px)

        # 3. Create the Premultiplied Overlays
        # We use 0.5 * alpha to ensure the mix totals 1.0 within the rod area
        left_overlay = img_l.astype(np.float32) * (alpha * 0.5)
        right_overlay = img_r.astype(np.float32) * (alpha * 0.5)

        # 4. Extract the Background (Original frame minus the rod)
        background = wide_roi_rgb.astype(np.float32) * (1.0 - alpha)

        # 5. Final Assembly: Background + Mixed Mirrored Content
        # Using np.clip to ensure no rounding errors push us past 255
        result = background + left_overlay + right_overlay
        # result = left_overlay * 2
        # result = right_overlay * 2
        return np.clip(result, 0, 255).astype(np.uint8)

    def inpaint_manual(self, wide_roi_rgb, blur_mask_u8):
        h, w, _ = wide_roi_rgb.shape

        for y in range(h-1, 0, -1):
            blur_row = blur_mask_u8[y, :]
            rgb_row = wide_roi_rgb[y, :]

            index255 = np.where(blur_row == 255)[0]
            if len(index255) == 0:
                break  # no more rod
            left255 = index255[0]
            right255 = index255[-1]
            rw = right255 - left255

            threshold = 1
            left0 = np.argmax(blur_row[:left255] > threshold)
            right0 = right255 + np.argmax(blur_row[right255:] <= threshold)
            w0 = right0 - left0

            # # Version A: copy L2-R2 mirrored as-is, no blur.
            # src_row = rgb_row[left255 - 1 : left255 - rw - 1 : -1, :]
            # rgb_row[left255:right255] = src_row[:]

            # Version B: copy L2-Ro mirrored on L2, as-is, no blur.
            # Note that we do NOT mirror the L0-L2 part as it _must_ contain the rod.
            src_row = rgb_row[left255 : 2*left255 - right0 : -1, :]
            rgb_row[left255 : right0] = src_row[:]

            # # Version C: same as B but apply blur mask in uint16 space
            mask_u8 = blur_row[left255 : right0, None].astype(np.uint16)
            src_row_u16 = src_row.astype(np.uint16)
            dst_row_u16 = rgb_row[left255 : right0].astype(np.uint16)
            # print(f"@@ mask_u8.shape={mask_u8.shape}, src_row_u16.shape={src_row_u16.shape}, dst_row_u16.shape={dst_row_u16.shape}")
            blended = (
                    dst_row_u16 * (255 - mask_u8)
                    + src_row_u16 * mask_u8
                ) // 255
            rgb_row[left255 : right0] = blended.astype(np.uint8)

            # # DEBUG
            # rgb_row[left0] = (255, 0, 0)
            # rgb_row[right0] = (255, 0, 0)
            # rgb_row[left255] = (0, 255, 0)
            # rgb_row[right255] = (0, 255, 0)

        return wide_roi_rgb


    def apply_masked_blur(self, rgb, mask_u8, ksize=(15, 15), sigma=0):
        """Applies Gaussian Blur to an image based on a gradient mask."""
        blurred_rgb = cv2.GaussianBlur(rgb, ksize, sigma)
        # Expand dims to (H, W, 1) so it broadcasts across R, G, and B
        alpha = mask_u8.astype(np.float32) / 255.0
        alpha = np.expand_dims(alpha, axis=-1)

        composite = (blurred_rgb.astype(np.float32) * alpha) + (rgb.astype(np.float32) * (1.0 - alpha))
        return composite.astype(np.uint8)

    def filter(self, frame_index, frame):
        if frame_index >= 0 and frame_index < len(self.frame_rods):
            rod = self.frame_rods[frame_index]
        else:
            print(f"@@ Detector: No rod info at frame {frame_index}")
            return frame

        if rod.isTunnel():
            return frame

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lu = lab[:, :, 0]

        # -- Phase 1
        # The rod mask is computed on a ROI of ROI_WIDTH, centered on the rod tracker x.
        # This helps with the temporal smooth as a ROI is a "view" centered on the rod
        # no matter where it is located in the image horizontally.

        rod_x_ctr = int(rod.center())
        roi_x_left, roi_lu, contrast_lu = self.extract_roi(lu, rod_x_ctr)
        if self.view_mask:
            self.draw_roi_bounds(roi_x_left, rod_x_ctr)

        roi_height = self.roi_height

        mask_u8 = self.find_rod_by_threshold(roi_lu,
            rod_x_ctr - roi_x_left,
            roi_height - 1)
        mask_u8 = self.keep_contiguous_rod(mask_u8,
            rod_x_ctr - roi_x_left,
            roi_height - 1)

        self.history_mask, mask_u8 = self.temporal_smooth_mask_u8(
            mask_u8, self.history_mask, weight_u8=256//4)
        mask_u8 = self.keep_contiguous_rod(mask_u8,
            rod_x_ctr - roi_x_left,
            roi_height - 1)

        # TBD we could detect (and skip) spurious invalid masks based
        # on pixel count jumping too high.

        # -- Phase 2
        # Starting form here, the ROI becomes the entire width of the image
        # and the bottom ROI height rows.
        wide_w = self.width
        wide_mask_u8 = np.zeros((roi_height, wide_w), np.uint8)
        wide_mask_u8[:, roi_x_left:roi_x_left + self.roi_width] = mask_u8

        wide_roi_rgb = frame[-roi_height:, :]

        # Original: Dilate by (1, h_dilate_width), blur by (h_dilate_width, 1)
        # Experiment: Dilate by (1, h_dilate_width), blur by (h_dilate_width, 1)
        h_dilate_width = 15
        kernel_h = np.ones((3, h_dilate_width), np.uint8)
        wide_mask_u8 = cv2.dilate(wide_mask_u8, kernel_h, iterations=1)
        h_blur_width = 15
        blur_mask_u8 = cv2.GaussianBlur(wide_mask_u8, (h_blur_width, 3), 0)

        if self.view_mask:
            self.draw_mask(blur_mask_u8, 0)

        if self.do_inpainting:
            # left0, right0 = self.measure_rod_width(blur_mask_u8,
            #     rod_x_ctr,
            #     roi_height - 1)
            # if right0 > left0:
            # inpainted = cv2.inpaint(wide_roi_rgb, blur_mask_u8, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
            # inpainted = cv2.inpaint(wide_roi_rgb, blur_mask_u8, inpaintRadius=5, flags=cv2.INPAINT_NS)
            # inpainted = self.apply_masked_blur(inpainted, blur_mask_u8, (1, 15))
            # h_blur_width = 15
            # blur_mask_u8 = cv2.GaussianBlur(wide_mask_u8, (h_blur_width, h_dilate_width), 0)

            #     # inpainted = self.inpaint_rod_biharmonic(roi_rgb, wide_mask_u8)

            #     inpainted = self.inpaint_dual_mirror(wide_roi_rgb, blur_mask_u8, left0, right0)
            inpainted = self.inpaint_manual(wide_roi_rgb, blur_mask_u8)
            frame[-roi_height:, :] = inpainted

        if self.view_mask:
            return cv2.cvtColor(lu, cv2.COLOR_GRAY2BGR)
        else:
            return frame

    def export(self):
        return super().export()

    def release(self):
        super().release()

