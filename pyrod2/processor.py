import cv2
import numpy as np

class ProcessorBase:
    def __init__(self):
        # Overlay is (B,G,R)
        self.overlay = None
        self.view_mask = False
        self.trigger_pause = False
        self.trigger_select_roi = False
        self.downscale = 1
        self.next_processor_requested = False
        self.compute_overlay = True

    def init_size(self, width, height):
        print(f"@@ {self} init_size")
        self.width = width
        self.height = height

    def init_overlay(self, frame):
        if self.overlay is None:
            self.overlay = np.zeros_like(frame)
        else:
            self.overlay[:] = (0, 0, 0)

    def combine_overlay(self, src):
        # This is a simplification of the RGB overlay blending where we
        # treat the source overlay as a binary threshold ... anything that is
        # not zero is copied over, without blending. It's less pretty as it
        # destroys text's and rod's alpha, but it's at least 2x faster.
        overlay = self.overlay
        mask = (overlay[..., 0] | overlay[..., 1] | overlay[..., 2]) > 0
        dest = src.copy()
        dest[mask] = overlay[mask]
        return dest

    def combine_overlay_unused(self, src_dst):
        # This is a "correct" RGB overlay blending on the RGB destination image.
        # It is quite slow and thus we don't use it. See combine_overlay() instead.
        gray_overlay = cv2.cvtColor(self.overlay, cv2.COLOR_BGR2GRAY)
        # Normalize grayscale to 0-1 range for alpha blending,
        # except we do all the operations in 16-bit integer arithmetics (for clipping).
        overlay_u16 = self.overlay.astype(np.uint16)
        src_dst_u16 = src_dst.astype(np.uint16)
        mask = np.clip(gray_overlay*3, 0, 255)[..., np.newaxis]
        blended = (
                src_dst_u16 * (255 - mask)
                + overlay_u16 * mask
            ) // 256
        return blended.astype(np.uint8)

    def select_roi(self, window_title, frame):
        pass

    def filter(self, frame_index, frame):
        return frame

    def pre_release(self):
        print(f"@@ {self} pre-release no-op")

    def export(self):
        print(f"@@ {self} export no-op")
        return {}

    def release(self):
        print(f"@@ {self} release no-op")
