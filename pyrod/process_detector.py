import cv2
import numpy as np
from processor import ProcessorBase
from rod import Rod

ROI_WIDTH = (4*35)/1280
ROI_HEIGHT = 5/12




class Detector(ProcessorBase):
    def __init__(self, locator):
        super().__init__()
        self.frame_rods = locator.frame_rods
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

    def init_size(self, width, height):
        super().init_size(width, height)
        print(f"@@ Detector init_size")
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
            (int(rod_x_ctr), self.height - 2),
            radius=5,
            color=(0, 255, 0),
            thickness=-1)


    def filter(self, frame_index, frame):
        if frame_index >= 0 and frame_index < len(self.frame_rods):
            rod = self.frame_rods[frame_index]
        else:
            return frame

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lu = lab[:, :, 0]

        rod_x_ctr = rod.center()
        roi_x_left, roi_lu, contrast_lu = self.extract_roi(lu, rod_x_ctr)
        self.draw_roi_bounds(roi_x_left, rod_x_ctr)

        return cv2.cvtColor(lu, cv2.COLOR_GRAY2BGR)

    def export(self, filename):
        super().export(filename)

    def release(self):
        super().release()

