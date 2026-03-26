import cv2
import numpy as np

class ProcessorBase:
    def __init__(self):
        # Overlay is (B,G,R)
        self.overlay = None
        self.trigger_pause = False

    def init_size(self, width, height):
        print(f"@@ {self} init_size")
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

    def filter(self, frame_index, frame):
        return frame

    def export(self, filename):
        print(f"@@ {self} export no-op")

    def release(self):
        print(f"@@ {self} release no-op")
