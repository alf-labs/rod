import cv2
import math
import numpy as np
import re
import scipy
from processor import ProcessorBase

class TopTracker(ProcessorBase):
    def __init__(self, start_frame):
        super().__init__()
        self.start_frame = start_frame

    def init_size(self, width, height):
        super().init_size(width, height)

    def init_overlay(self, frame):
        super().init_overlay(frame)

    def filter(self, frame_index, frame):
        return super().filter(frame_index, frame)

    def export(self):
        return super().export()

    def release(self):
        super().release()

