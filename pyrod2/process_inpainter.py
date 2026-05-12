import cv2
import numpy as np
import scipy
from rect import Rect
from processor import ProcessorBase
from process_coupler import QUALITY_THRESHOLD

ROI_WIDTH_PCT = 1/2
ROD_BLUR_PY = 40/720
COUPLER_MULTIPLER = 4
ROD_DILATE_PX = 10
ROD_BLUR_PX = 21

class ProcessInpainter(ProcessorBase):
    def __init__(self, coupler_tracker, rod_detector, inpainting="left", rod_dilate_px=ROD_DILATE_PX, rod_blur_px=ROD_BLUR_PX):
        super().__init__()
        self.coupler_tracker = coupler_tracker
        self.rod_detector = rod_detector
        self.tracker_templates = coupler_tracker.tracker_templates
        self.couplers = coupler_tracker.couplers
        self.current_template = None
        self.roi_rect = None
        self.view_mask = inpainting is None
        self.rod_dilate_px = rod_dilate_px
        self.rod_blur_px = rod_blur_px
        self.inpaint_method = {
            "left":   self.inpaint_poly_left,
            "right":  self.inpaint_poly_right,
            "mix":    self.inpaint_poly_mix,
            "none":   self.inpaint_noop,
        }.get(inpainting, None)
        self.rod_h_top = None
        print(f"@@ Inpainter method: {self.inpaint_method}")

    def init_size(self, width, height):
        super().init_size(width, height)
        print(f"@@ Inpainter init_size")
        self.roi_center = width / 2
        self.rod_blur_py = int(ROD_BLUR_PY * height)
        print(f"Inpainter: Rod blur height {self.rod_blur_py} px")
        self.rod_blend_x_u16 = self.smoothstep(self.rod_blur_px)
        print(f"@@ Inpainter view_mask: {self.view_mask}")

    def init_overlay(self, frame):
        super().init_overlay(frame)

    def smoothstep(self, width_px):
        # https://en.wikipedia.org/wiki/Smoothstep
        # Note: in the [0..1] range, smoothstep[1-x] = 1-smoothstep[x]
        # meaning to "reverse" the curve for blending, we can either
        # use "1-s(x)" or "s(1-x)".
        xs = np.arange(width_px) / width_px
        return (xs * xs * (3.0 - 2.0 * xs) * 256).astype(np.uint16)

    def filter(self, window_info, frame_index, frame):
        h, w = frame.shape[:2]
        coupler = self.coupler_tracker.couplers[frame_index]
        rod = self.rod_detector.rods[frame_index]
        if rod is None or coupler is None or coupler.quality < QUALITY_THRESHOLD:
            return frame

        if self.current_template is None:
            self.current_template = self.coupler_tracker.tracker_templates[coupler.coupler_ref]
            print(f"@@ Tracker size: {self.current_template.rect}")
            self.rod_blur_py = min(self.rod_blur_py, self.current_template.rect.h)
        coupler_template = self.current_template
        # CR: a rect centered on current coupler position, of same w/h as the coupler template.
        cr = coupler_template.rect.copy()
        cr.recenter_to(coupler.center.x, coupler.center.y)

        # The overall ROI window (from top of static coupler template to bottom of video)
        roi_rect = self.get_search_window(cr)
        ry1 = roi_rect.y
        ry2 = ry1 + roi_rect.h
        rx1 = roi_rect.x
        rx2 = rx1 + roi_rect.w

        roi_rgb = frame[ry1 : ry2, rx1 : rx2]

        if self.inpaint_method:
            # print(f"@@ ----- frame [{frame_index:04d}] ------")
            inpainted = self.inpaint_method(roi_rect, roi_rgb, rod)
            frame[ry1 : ry2, rx1 : rx2] = inpainted

        if self.view_mask and self.compute_overlay:
            self.draw_rect(cr,       (255, 128, 0))
            self.draw_rect(roi_rect, (255, 255, 0))
            self.draw_rod (roi_rect, rod)

        return frame

    def get_search_window(self, coupler_rect):
        width = self.width
        height = self.height
        c = coupler_rect.center()
        w = int(ROI_WIDTH_PCT * width)
        if self.roi_rect is None:
            y = coupler_rect.y - coupler_rect.h
        else:
            y = self.roi_rect.y
        x = int(self.roi_center - w // 2)
        h = height - y
        self.roi_center = self.roi_center * 0.25 + c[0] * 0.75
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

    def draw_rect(self, rect, color, width=2):
        x = rect.x
        y = rect.y
        w = rect.w
        h = rect.h
        cv2.rectangle(self.overlay, (x, y), (x + w - 1, y + h - 1), color, width)

    def draw_rod(self, roi_rect, rod):
        ys = np.linspace(roi_rect.y, roi_rect.y + roi_rect.h, num=10)

        xc = rod.poly_c(ys)
        xw = rod.poly_w(ys)

        x1s = xc - xw / 2
        x2s = xc + xw / 2
        self._draw_poly(x1s, ys, (0, 255, 255))
        self._draw_poly(x2s, ys, (0, 255, 255))

        x1s -= self.rod_dilate_px
        x2s += self.rod_dilate_px
        self._draw_poly(x1s, ys, (0, 0, 255))
        self._draw_poly(x2s, ys, (0, 0, 255))

        x1s -= self.rod_blur_px
        x2s += self.rod_blur_px
        self._draw_poly(x1s, ys, (0, 255, 0))
        self._draw_poly(x2s, ys, (0, 255, 0))

    def _draw_poly(self, xs, ys, color):
        # Format the points for OpenCV and draw polyline
        # Points must be (x, y) integers in a shape of (N, 1, 2)
        pts = np.column_stack((xs, ys)).astype(np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(self.overlay, [pts], isClosed=False, color=color, thickness=1, lineType=cv2.LINE_AA)

    def export(self):
        return super().export()

    def release(self):
        super().release()

    def inpaint_noop(self, roi_rect, roi_rgb, rod):
        return roi_rgb

    def inpaint_poly_left(self, roi_rect, roi_rgb, rod):
        return self._inpaint_poly_left_right(roi_rect, roi_rgb, rod, self._left_merge)

    def inpaint_poly_right(self, roi_rect, roi_rgb, rod):
        return self._inpaint_poly_left_right(roi_rect, roi_rgb, rod, self._right_merge)

    def inpaint_poly_mix(self, roi_rect, roi_rgb, rod):
        return self._inpaint_poly_left_right(roi_rect, roi_rgb, rod, self._mixed_merge)

    def _inpaint_poly_left_right(self, roi_rect, roi_rgb, rod, method):
        ry1 = roi_rect.y
        ry2 = ry1 + roi_rect.h
        rx1 = roi_rect.x
        rx2 = rx1 + roi_rect.w

        ry_top = rod.y_top # - self.rod_blur_py
        for y1 in range(ry_top, ry2):
            lc = rod.poly_c(y1) - rx1
            lw = rod.poly_w(y1)
            method(lc, lw, y1 - ry1, roi_rgb)

        py = self.rod_blur_py
        if py > 0:
            lc1 = rod.poly_c(ry_top) - rx1
            lw1 = rod.poly_w(ry_top)
            for y1 in range(ry_top - py, ry_top):
                t = (ry_top - y1) / py  # from 1 (top) to 0 (bottom, by ry_top)
                coef = int(256 - t * 256)
                if coef <= 0:
                    continue
                lw = lw1 + lw1 * COUPLER_MULTIPLER * t
                ly = y1 - ry1
                if coef < 256:
                    org_row_u16 = roi_rgb[ly, :].copy().astype(np.uint16)
                # print(f"@@ -----     y [{y1: 3d}] ------")
                method(lc1, lw, ly, roi_rgb)
                if coef < 256:
                    new_row_u16 = roi_rgb[ly, :].astype(np.uint16)
                    blended = (
                        new_row_u16 * coef
                        + org_row_u16 * (256 - coef)
                    ) / 256
                    roi_rgb[ly, :] = blended.astype(np.uint8)

        return roi_rgb

    def _left_merge(self, lc, lw, ly, roi_rgb):
        # X values:
        # x0 --> blend (w0) --> x1 (left) --> full (w1) --> x2 (right) --> blend (w2) --> x3
        # Left  algorithm: no blend on left (x0..x1), blend on the right (x2..x3)
        # Right algorithm: blend on the left (x0..x1), no blend on right (x2..x3), only
        x1 = int(lc - lw / 2) - self.rod_dilate_px
        x2 = int(lc + lw / 2) + self.rod_dilate_px
        x0 = x1 - self.rod_blur_px
        x3 = x2 + self.rod_blur_px
        w0 = x1 - x0
        w1 = x2 - x1
        w2 = x3 - x2

        rgb_row = roi_rgb[ly, :]

        # Part 1: copy X1-X2 mirrored around X1 as-is, no blend.
        src_row = rgb_row[x1 : x1 - w1 : -1, :]
        rgb_row[x1 : x2] = src_row[:]

        # Part 2: blend X2-X3 mirrored around X1.
        src_row_u16 = rgb_row[x1 - w1 : x1 - w1 - w2 : -1, :].astype(np.uint16)
        dst_row_u16 = rgb_row[x2 : x3].astype(np.uint16)
        blend_u16 = self.rod_blend_x_u16
        # print(f"@@ lc {lc}, lw {lw}, x0 {x0} + {w0} > x1 {x1} + {w1} > x2 {x2} + {w2} > x3 {x3} -- rgb {rgb_row.shape}, src {src_row_u16.shape}, dst {dst_row_u16.shape}, blend {blend_u16.shape}")
        blended = (
                  dst_row_u16 * blend_u16[:   , np.newaxis]
                + src_row_u16 * blend_u16[::-1, np.newaxis]
            ) / 256
        rgb_row[x2 : x3] = blended.astype(np.uint8)

    def _right_merge(self, lc, lw, ly, roi_rgb):
        # X values:
        # x0 --> blend (w0) --> x1 (left) --> full (w1) --> x2 (right) --> blend (w2) --> x3
        # Left  algorithm: no blend on left (x0..x1), blend on the right (x2..x3)
        # Right algorithm: blend on the left (x0..x1), no blend on right (x2..x3), only
        x1 = int(lc - lw / 2) - self.rod_dilate_px
        x2 = int(lc + lw / 2) + self.rod_dilate_px
        x0 = x1 - self.rod_blur_px
        x3 = x2 + self.rod_blur_px
        w0 = x1 - x0
        w1 = x2 - x1
        w2 = x3 - x2

        rgb_row = roi_rgb[ly, :]

        # Part 1: copy X1-X2 mirrored around X2 as-is, no blend.
        src_row = rgb_row[x2 + w1 : x2 : -1, :]
        rgb_row[x1 : x2] = src_row[:]

        # Part 2: blend X0-X1 mirrored around X2.
        src_row_u16 = rgb_row[x2 + w1 + w0 : x2 + w1 : -1, :].astype(np.uint16)
        dst_row_u16 = rgb_row[x0 : x1].astype(np.uint16)
        blend_u16 = self.rod_blend_x_u16
        # print(f"@@ lc {lc}, lw {lw}, x0 {x0} + {w0} > x1 {x1} + {w1} > x2 {x2} + {w2} > x3 {x3} -- rgb {rgb_row.shape}, src {src_row_u16.shape}, dst {dst_row_u16.shape}, blend {blend_u16.shape}")
        blended = (
                  dst_row_u16 * blend_u16[::-1, np.newaxis]
                + src_row_u16 * blend_u16[:   , np.newaxis]
            ) / 256
        rgb_row[x0 : x1] = blended.astype(np.uint8)

    def _mixed_merge(self, lc, lw, ly, roi_rgb):
        # X values:
        # x0 --> blend (w0) --> x1 (left) --> full (w1) --> x2 (right) --> blend (w2) --> x3
        # Left  algorithm: no blend on left (x0..x1), blend on the right (x2..x3)
        # Right algorithm: blend on the left (x0..x1), no blend on right (x2..x3), only
        x1 = int(lc - lw / 2) - self.rod_dilate_px
        x2 = int(lc + lw / 2) + self.rod_dilate_px
        x0 = x1 - self.rod_blur_px
        x3 = x2 + self.rod_blur_px
        w0 = x1 - x0
        w1 = x2 - x1
        w2 = x3 - x2

        # We need to copy the source in order to not read what we just overwrote
        rgb_row = roi_rgb[ly, :].copy()

        # Part 1: copy X1-X2 mirrored around X2 as-is, no blend.
        src_row1 = rgb_row[x1 : x1 - w1 : -1, :].astype(np.uint16)
        src_row2 = rgb_row[x2 + w1 : x2 : -1, :].astype(np.uint16)
        blended = (src_row1 + src_row2) // 2
        roi_rgb[ly, x1 : x2] = blended.astype(np.uint8)

        # Part 2: blend X2-X3 mirrored around X1.
        src_row_u16 = rgb_row[x1 - w1 : x1 - w1 - w2 : -1, :].astype(np.uint16)
        dst_row_u16 = rgb_row[x2 : x3].astype(np.uint16)
        blend_u16 = self.rod_blend_x_u16
        blended = (
                  dst_row_u16 * blend_u16[:   , np.newaxis]
                + src_row_u16 * blend_u16[::-1, np.newaxis]
            ) // 256
        roi_rgb[ly, x2 : x3] = blended.astype(np.uint8)

        # Part 2: blend X0-X1 mirrored around X2.
        src_row_u16 = rgb_row[x2 + w1 + w0 : x2 + w1 : -1, :].astype(np.uint16)
        dst_row_u16 = rgb_row[x0 : x1].astype(np.uint16)
        blend_u16 = self.rod_blend_x_u16
        blended = (
                  dst_row_u16 * blend_u16[::-1, np.newaxis]
                + src_row_u16 * blend_u16[:   , np.newaxis]
            ) // 256
        roi_rgb[ly, x0 : x1] = blended.astype(np.uint8)
