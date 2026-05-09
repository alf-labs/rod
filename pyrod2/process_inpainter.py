import cv2
import numpy as np
import scipy
from rect import Rect
from processor import ProcessorBase
from process_coupler import ROI_WIDTH_PCT, QUALITY_THRESHOLD

# ROI_WIDTH_MULTIPLIER = 5 # x rod_width_px
# ROI_HEIGHT = 400/720
ROD_BLUR_PY = 11/720
ROD_DILATE_PX = 3
ROD_BLUR_PX = 3

class ProcessInpainter(ProcessorBase):
    def __init__(self, coupler_tracker, rod_detector, inpainting="left", rod_dilate_px=ROD_DILATE_PX, rod_blur_px=ROD_BLUR_PX):
        super().__init__()
        self.coupler_tracker = coupler_tracker
        self.rod_detector = rod_detector
        self.tracker_templates = coupler_tracker.tracker_templates
        self.couplers = coupler_tracker.couplers
        self.current_template = None
        self.view_mask = inpainting is None
        self.inpaint_method = {
            "left":   self.inpaint_manual_left,
            "right":  self.inpaint_manual_right,
            "mix":    self.inpaint_manual_mix,
            "telea":  self.inpaint_telea,
            "navier": self.inpaint_navier,
            "none":   self.inpaint_noop,
        }.get(inpainting, None)
        self.rod_dilate_kernel = np.ones((3, rod_dilate_px), np.uint8)
        self.rod_blur_ksize = (rod_blur_px, 3)

    def init_size(self, width, height):
        super().init_size(width, height)
        print(f"@@ Inpainter init_size")
        # self.rod_blur_py = int(ROD_BLUR_PY * height)
        # print(f"Inpainter: Rod {self.rod_width_px} px")

    def init_overlay(self, frame):
        super().init_overlay(frame)

    # def extract_roi(self, lu, rod_x_ctr):
    #     roi_x_left  = int(rod_x_ctr - self.roi_width / 2)
    #     roi_x_right = roi_x_left + self.roi_width

    #     roi_lu = lu[-self.roi_height:, roi_x_left:roi_x_right].copy()

    #     # Do not run the CLAHE and contrast Histogram Stretching (Min-Max Normalization),
    #     # it ruins the mask more than it helps.

    #     return roi_x_left, roi_lu

    # def draw_roi_bounds(self, roi_x_left, rod_x_ctr):
    #     cv2.rectangle(self.overlay,
    #         (roi_x_left, self.height - self.roi_height),
    #         (roi_x_left + self.roi_width, self.height - 1),
    #         color=(255, 255, 0),
    #         thickness=1)

    #     cv2.circle(self.overlay,
    #         (rod_x_ctr, self.height - 2),
    #         radius=5,
    #         color=(0, 255, 0),
    #         thickness=-1)

    # def weight(self, a, b, weight_a=0.75):
    #     return a * weight_a + b * (1 - weight_a)

    # def smoothstep(self, x):
    #     """x is expected in range 0..1 as float"""
    #     # https://en.wikipedia.org/wiki/Smoothstep
    #     if x < 0:
    #         return 0
    #     elif x >= 1:
    #         return 1
    #     else:
    #         return x * x * (3.0 - 2.0 * x)

    # def find_rod_by_threshold(self, roi_lu, tracked_x, tracked_y):
    #     # Try a basic binary mask
    #     mean_luminance = int(np.mean(roi_lu))
    #     # median_luminance = int(np.median(roi_lu))
    #     tracked_luminance = int(roi_lu[tracked_y, tracked_x])
    #     trigger_luminance = (mean_luminance + tracked_luminance * 2) // 3
    #     mask_b = roi_lu >= trigger_luminance
    #     mask_u8 = mask_b.astype(np.uint8) * 255

    #     # print(f"""@@ lum mean {mean_luminance} < median {median_luminance} < tracked {tracked_luminance} --> trigger {trigger_luminance} :
    #     # @@ {roi_lu[roi_lu.shape[0]//2, :]}
    #     # @@ {mask_u8[roi_lu.shape[0]//2, :]}""")

    #     # Erode and dilate
    #     # Kernel choices: 3x3 typical, or 5x1 (vertical band) to favor vertical features.
    #     kernel = np.ones((5, 3), np.uint8)
    #     mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)

    #     # Erode the mask horizontally a bit more
    #     kernel_h = np.ones((11, 15), np.uint8)
    #     mask_u8 = cv2.dilate(mask_u8, kernel_h, iterations=1)

    #     return mask_u8

    # def temporal_smooth_mask_u8(self, mask_u8, history_u8, weight_u8):
    #     if history_u8 is None:
    #         return mask_u8, mask_u8
    #     inv_weight_u8 = 256 - weight_u8

    #     # Formula: (History * (256 - alpha) + New * alpha) / 256
    #     acc = (history_u8.astype(np.uint16) * inv_weight_u8) + (mask_u8.astype(np.uint16) * weight_u8)
    #     # Bit-shift to divide by 256
    #     new_history = (acc >> 8).astype(np.uint8)

    #     # Original 0.75 threshold becomes 191 (0.75 * 255)
    #     # cv2.threshold is much faster than numpy boolean comparisons
    #     _, mask_u8 = cv2.threshold(new_history, 191, 255, cv2.THRESH_BINARY)

    #     return new_history, mask_u8

    # def keep_contiguous_rod(self, mask_u8, seed_x, seed_y):
    #     h, w = mask_u8.shape
    #     flood_mask = np.zeros((h + 2, w + 2), np.uint8) # floodFill needs +2 size

    #      # Note: use FLOODFILL_MASK_ONLY to avoid modifying mask_f32 (we don't nede it)
    #     # thus this ignores newVal. Just kept it for the sake of example.
    #     cv2.floodFill(
    #         mask_u8,
    #         flood_mask,
    #         (seed_x, seed_y),
    #         newVal=255,
    #         flags=cv2.FLOODFILL_MASK_ONLY)

    #     return flood_mask[1:-1, 1:-1] * 255

    # def draw_mask_heatmap(self, mask, roi_x_left):
    #     height, width = mask.shape
    #     overlay_view = self.overlay[-height:, roi_x_left:roi_x_left + width]
    #     # overlay_view[mask >    1] = (0, 255, 255)
    #     # overlay_view[mask >  128] = (0, 128, 255)
    #     # overlay_view[mask == 255] = (0,   0, 255)
    #     # Fill the overlay with (0, mask, 255)
    #     heatmap = overlay_view.copy()
    #     heatmap[:, :, 0] = 0
    #     heatmap[:, :, 1] = mask
    #     heatmap[:, :, 2] = 255
    #     overlay_view[mask > 0] = heatmap[mask > 0]

    # def draw_mask_outline(self, mask_u8, roi_x_left):
    #     h, w = mask_u8.shape
    #     overlay_view = self.overlay[-h:, roi_x_left:roi_x_left + w]

    #     rows = np.arange(h)     # all rows as indices [0...h-1]

    #     # argmax returns the index of the FIRST True value it encounters
    #     # axis=1 means to accross axis 1 (which is W in the H,W order)
    #     start0 = np.argmax(mask_u8 > 0, axis=1)
    #     start2 = np.argmax(mask_u8 == 255, axis=1)
    #     flip_u8 = mask_u8[:, ::-1]  # step -1 mirrors on W axis
    #     end0 = (w - 1) - np.argmax(flip_u8 > 0, axis=1)
    #     end2 = (w - 1) - np.argmax(flip_u8 == 255, axis=1)

    #     overlay_view[rows, start0] = (0, 255, 255)
    #     overlay_view[rows, start2] = (0,   0, 255)
    #     overlay_view[rows, end2  ] = (0,   0, 255)
    #     overlay_view[rows, end0  ] = (0, 255, 255)

    # def draw_mask_line(self, mask_u8, roi_x_left, y_mask):
    #     _, w = mask_u8.shape
    #     overlay_view = self.overlay[:, roi_x_left:roi_x_left + w]
    #     h, _, _ = overlay_view.shape

    #     color = (255, 0, 0)
    #     line_u8 = mask_u8[y_mask, :]

    #     lx = 0
    #     ly = int(line_u8[0])
    #     dy = int(h - 10)
    #     for x in range(1, w):
    #         curr = int(line_u8[x])
    #         cv2.line(overlay_view, (lx, dy - ly), (x, dy - curr), color, 2)
    #         lx = x
    #         ly = curr

    def inpaint_telea(self, wide_roi_rgb, blur_mask_u8):
        return cv2.inpaint(wide_roi_rgb, blur_mask_u8, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    def inpaint_navier(self, wide_roi_rgb, blur_mask_u8):
        return cv2.inpaint(wide_roi_rgb, blur_mask_u8, inpaintRadius=5, flags=cv2.INPAINT_NS)

    def inpaint_noop(self, wide_roi_rgb, blur_mask_u8):
        return wide_roi_rgb

    def update_rod_h_top(self, y):
        if self.rod_h_top is None:
            self.rod_h_top = int(y)
        else:
            self.rod_h_top = int(self.weight(self.rod_h_top, y, 0.1))

    def inpaint_manual_left(self, wide_roi_rgb, blur_mask_u8, mask_transform=None):
        h, w, _ = wide_roi_rgb.shape

        if self.rod_h_top is None or self.rod_blur_py == 0:
            y_blur_0 = 0
            y_blur_1 = 0
        else:
            y_blur_1 = self.rod_h_top
            y_blur_0 = y_blur_1 + self.rod_blur_py

        for y in range(h-1, 0, -1):
            blur_row = blur_mask_u8[y, :]
            rgb_row = wide_roi_rgb[y, :]

            index255 = np.where(blur_row == 255)[0]
            if len(index255) == 0:
                self.update_rod_h_top(y)
                break  # no more rod
            left2 = index255[0]
            right2 = index255[-1]
            w2 = right2 - left2

            threshold = 1
            left0 = np.argmax(blur_row[:left2] > threshold)
            right0 = right2 + np.argmax(blur_row[right2:] <= threshold)
            w0 = right0 - left0

            line_blend = 1
            if y < y_blur_0:
                dy = y_blur_0 - y
                line_blend = 1 - self.smoothstep(dy / (y_blur_0 - y_blur_1))
            line_blend = int(256 * line_blend)

            # "left" means we just mirror columns on the left (L2) part of the rod.
            # Destination: L2 -> (plateau) R2 (a W2 width) --> (gradient) R0
            # A mirror on L2: (x) -> 2*L2-x
            # Source: L2 -> (plateau) 2*L2-R2 --> (gradient) -> 2*L2 - R0, step -1.

            # # Version A: copy L2-R2 mirrored as-is, no blur (same as right).
            # src_row = rgb_row[left2 : left2 - w2 : -1, :]
            # Version A: rgb_row[left2:right2] = src_row[:]

            # Version B: copy L2-R0 mirrored on L2, as-is, no blur.
            # Note that we do NOT mirror the L0-L2 part as it _must_ contain the rod.
            src_row = rgb_row[left2 : 2*left2 - right0 : -1, :]
            # Version B: rgb_row[left2 : right0] = src_row[:]

            # # Version C: same as B but apply blur mask in uint16 space
            mask_u8 = blur_row[left2 : right0, None].astype(np.uint16)
            if mask_transform:  # see "mix" version
                mask_u8 = mask_transform(mask_u8)
            src_row_u16 = src_row.astype(np.uint16)
            dst_row_u16 = rgb_row[left2 : right0].astype(np.uint16)
            line_mask_u16 = (mask_u8.astype(np.uint16) * line_blend) // 256
            blended = (
                    dst_row_u16 * (255 - line_mask_u16)
                    + src_row_u16 * line_mask_u16
                ) // 255
            rgb_row[left2 : right0] = blended.astype(np.uint8)

        return wide_roi_rgb

    def inpaint_manual_right(self, wide_roi_rgb, blur_mask_u8, mask_transform=None):
        h, w, _ = wide_roi_rgb.shape

        if self.rod_h_top is None or self.rod_blur_py == 0:
            y_blur_0 = 0
            y_blur_1 = 0
        else:
            y_blur_1 = self.rod_h_top
            y_blur_0 = y_blur_1 + self.rod_blur_py

        for y in range(h-1, 0, -1):
            blur_row = blur_mask_u8[y, :]
            rgb_row = wide_roi_rgb[y, :]

            index255 = np.where(blur_row == 255)[0]
            if len(index255) == 0:
                self.update_rod_h_top(y)
                break  # no more rod
            left2 = index255[0]
            right2 = index255[-1]
            w2 = right2 - left2

            threshold = 1
            left0 = np.argmax(blur_row[:left2] > threshold)
            right0 = right2 + np.argmax(blur_row[right2:] <= threshold)
            w0 = right0 - left0

            line_blend = 1
            if y < y_blur_0:
                dy = y_blur_0 - y
                line_blend = 1 - self.smoothstep(dy / (y_blur_0 - y_blur_1))
            line_blend = int(256 * line_blend)

            # "right" means we just mirror columns on the right (R2) part of the rod.
            # Destination: L0 (gradient) -> L2 -> (plateau) R2 (a W2 width)
            # A mirror on R2: (x) -> 2*R2-x
            # Source: 2*R2 - L0 (gradient) -> 2*R2 - L2 (plateau) --> (gradient) R2, step -1.

            # # Version A: copy L2-R2 mirrored as-is, no blur (same as left).
            # src_row = rgb_row[left2 : left2 - w2 : -1, :]
            # Version A: rgb_row[left2:right2] = src_row[:]

            # Version B: copy L0-R2 mirrored on R2, as-is, no blur.
            # Note that we do NOT mirror the R2->R0 part as it _must_ contain the rod.
            src_row = rgb_row[2*right2 - left0 : right2 : -1, :]
            # Version B: rgb_row[left0 : right2] = src_row[:]

            # # Version C: same as B but apply blur mask in uint16 space
            mask_u8 = blur_row[left0 : right2, None].astype(np.uint16)
            if mask_transform:  # see "mix" version
                mask_u8 = mask_transform(mask_u8)
            src_row_u16 = src_row.astype(np.uint16)
            dst_row_u16 = rgb_row[left0 : right2].astype(np.uint16)
            line_mask_u16 = (mask_u8.astype(np.uint16) * line_blend) // 256
            blended = (
                    dst_row_u16 * (255 - line_mask_u16)
                    + src_row_u16 * line_mask_u16
                ) // 255
            rgb_row[left0 : right2] = blended.astype(np.uint8)

        return wide_roi_rgb

    def inpaint_manual_mix(self, wide_roi_rgb, blur_mask_u8):
        # "mix" means we do mirror from the left *and* the right, and then average:
        # We take both the L0 gradient from the "left" algorithm,
        # and the R0 gradient from the "right" algorithm,
        # however the middle L2->R2 part is a plateau averaging left+right.
        # (another variation is to treat is a cross-over gradient, but that implies more copies)

        # The trick used is that we reuse the left/right inpainting but just before
        # applying the uint16 mask we transform it by changing the plateau 255 values to 127.
        # This halves the plateau values and combines both left and right.
        # (technicall 127*2=254 so we loose 1/256th luminosity)
        # We cannot do that upfront on blur_mask_u8 as we need the 255 values intact for the
        # left/right boundary detection.

        def mask_transform(mask_u16):
            mask_u16[mask_u16 == 255] = 127
            return mask_u16
        wide_roi_rgb = self.inpaint_manual_left(wide_roi_rgb, blur_mask_u8, mask_transform)
        wide_roi_rgb = self.inpaint_manual_right(wide_roi_rgb, blur_mask_u8, mask_transform)
        return wide_roi_rgb

    def apply_masked_blur(self, rgb, mask_u8, ksize=(15, 15), sigma=0):
        """Applies Gaussian Blur to an image based on a gradient mask."""
        blurred_rgb = cv2.GaussianBlur(rgb, ksize, sigma)
        # Expand dims to (H, W, 1) so it broadcasts across R, G, and B
        alpha = mask_u8.astype(np.float32) / 255.0
        alpha = np.expand_dims(alpha, axis=-1)

        composite = (blurred_rgb.astype(np.float32) * alpha) + (rgb.astype(np.float32) * (1.0 - alpha))
        return composite.astype(np.uint8)

    def filter(self, window_title, frame_index, frame):
        h, w = frame.shape[:2]
        coupler = self.coupler_tracker.couplers[frame_index]
        rod = self.rod_detector.rods[frame_index]
        if rod is None or coupler is None or coupler.quality < QUALITY_THRESHOLD:
            return frame

        if self.current_template is None:
            self.current_template = self.coupler_tracker.tracker_templates[coupler.coupler_ref]
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

        if self.compute_overlay and rod is not None:
            self.draw_rod(rod, (0, 255, 255), width=1)

        return frame

        # -- Phase 2
        # Starting form here, the ROI becomes the entire width of the image
        # and the bottom ROI height rows.
        wide_w = self.width
        wide_mask_u8 = np.zeros((roi_height, wide_w), np.uint8)
        wide_mask_u8[:, roi_x_left:roi_x_left + self.roi_width] = mask_u8

        wide_roi_rgb = frame[-roi_height:, :]

        # Original: Dilate by (1, h_dilate_width), blur by (h_dilate_width, 1)
        # Experiment: Dilate by (1, h_dilate_width), blur by (h_dilate_width, 1)
        wide_mask_u8 = cv2.dilate(wide_mask_u8, self.rod_dilate_kernel, iterations=1)
        blur_mask_u8 = cv2.GaussianBlur(wide_mask_u8, self.rod_blur_ksize, 0)

        if self.view_mask and self.compute_overlay:
            self.draw_mask_heatmap(blur_mask_u8, 0)
            # self.draw_mask_outline(blur_mask_u8, 0)
            # self.draw_mask_line(blur_mask_u8, 0, -10)

        if self.inpaint_method:
            inpainted = self.inpaint_method(wide_roi_rgb, blur_mask_u8)
            frame[-roi_height:, :] = inpainted

        if self.view_mask and self.compute_overlay:
            self.draw_roi_bounds(roi_x_left, rod_x_ctr)

        if self.view_mask:
            return cv2.cvtColor(lu, cv2.COLOR_GRAY2BGR)
        else:
            return frame

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

    def export(self):
        return super().export()

    def release(self):
        super().release()

