#!/usr/bin/python
# Ensure we're running from the Virtual Env version
import os
if not "VIRTUAL_ENV" in os.environ:
    print("ERROR: Run this from venv using 'source ./venv_catd/bin/activate' first")
    exit(1)
IS_RPI = os.path.isfile("/etc/rpi-issue")

import base64
import sys
import time

try:
    import cv2
    import numpy as np
    import imutils
    from flask import Flask, render_template, Response, request, jsonify
except ModuleNotFoundError as e:
    print(f"ERROR: Missing library. {e}")
    print( "To fix: $ pip install opencv-python numpy imutils flask")
    print(f"or    : $ python {sys.argv[0]}")
    exit(1)

WINDOW_TITLE = "Rod Sample"
VIDEOS = [
    "../samples/rod1_front_randall_up_2025-03-23.mp4",
    "../samples/rod1_rear_randall_up_2025-03-23.mp4",
]

FPS = 30
FPS_MS = 1000//FPS


def detect_edges_sobel_canny(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 1. Focus on vertical edges using Sobel X
    # ddepth=cv2.CV_64F helps catch the transition from light to dark and vice versa
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    sobel_8bit = np.uint8(abs_sobelx)
    
    # 2. Canny for cleaner lines
    edges = cv2.Canny(sobel_8bit, 50, 150)
    
    # 3. Region of Interest (ROI) - Bottom Center
    h, w = edges.shape
    roi_mask = np.zeros_like(edges)
    cv2.rectangle(roi_mask, (int(w*0.4), int(h*0.5)), (int(w*0.6), h), 255, -1)
    masked_edges = cv2.bitwise_and(edges, roi_mask)
    
    return masked_edges


def draw_line(source, channel_index, y, color, dest):
    
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
    dy = h - 10
    for x in range(1, w):
        curr = cvalue(x)
        cv2.line(dest, (lx, dy - ly), (x, dy - curr), color, 2)
        lx = x
        ly = curr


def detect_by_color_lab(frame):
    # image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # channel = lab[:, :, 0] # L

    # mask = cv2.inRange(channel, 140, 160)
    # result = cv2.bitwise_and(frame, frame, mask=mask)

    # rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


    lu, au, bu = cv2.split(lab)     # uint8
    aS = au.astype(np.int16) - 128
    bS = bu.astype(np.int16) - 128
    ab_diff = np.abs(aS - bS)
    ab_diff_output = ab_diff.astype(np.uint8)
    # result = cv2.cvtColor(ab_diff_output, cv2.COLOR_GRAY2BGR)

    # # L, a, b filter
    # lower_lab = np.array([128, 125, 125])
    # upper_lab = np.array([255, 131, 131])
    # mask = cv2.inRange(lab, lower_lab, upper_lab)
    # result = cv2.bitwise_and(frame, frame, mask=mask)

    # L filter
    mask = cv2.inRange(lu, 128, 170)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    # a-b filter
    mask = cv2.inRange(ab_diff_output, 0, 4)
    result = cv2.bitwise_and(result, result, mask=mask)

    draw_line(ab_diff_output, 0, -1, (0, 255, 255), result)
    draw_line(lab, 0, -1, (0, 0, 255), result)
    # draw_line(lab, 1, -1, (0, 255, 255), result)
    # draw_line(lab, 2, -1, (255, 0, 255), result)

    return result

def detect_by_color_hsv(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define 'Gray' range: Low saturation, mid-to-high value
    # Hue doesn't matter much for gray, so we take the full range (0-180)
    lower_gray = np.array([0, 0, 50])   
    upper_gray = np.array([180, 50, 200])
    
    mask = cv2.inRange(hsv, lower_gray, upper_gray)
    
    # Cleanup noise with Morphological Opening (Erosion followed by Dilation)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask


def detect_hough_lines(frame):
    # Start with the edge detection from method 1
    edges = detect_edges_sobel_canny(frame)
    
    # Probabilistic Hough Transform
    # rho=1, theta=pi/180, threshold=50, minLineLength=100, maxLineGap=10
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, 
                            minLineLength=100, maxLineGap=20)
    
    line_img = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate angle: We only want vertical lines (approx 90 degrees)
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            if 70 < angle < 110: # Vertical tolerance
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
    return line_img


class RodRemover:
    def __init__(self, alpha=0.8, inpaint_radius=3):
        self.alpha = alpha  # Persistence of the mask (0.0 to 1.0)
        self.inpaint_radius = inpaint_radius
        self.running_mask = None

    def get_rod_mask(self, frame):
        h, w = frame.shape[:2]
        # 1. Convert to HSV for robust color segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define 'Neutral Gray' (Low saturation, specific value range)
        lower_gray = np.array([0, 0, 40])
        upper_gray = np.array([180, 60, 200])
        mask = cv2.inRange(hsv, lower_gray, upper_gray)

        # 2. Focus on the Bottom/Center ROI (where the rod starts)
        roi_mask = np.zeros_like(mask)
        # Search the bottom 70% of the frame
        cv2.rectangle(roi_mask, (0, int(h * 0.3)), (w, h), 255, -1)
        mask = cv2.bitwise_and(mask, roi_mask)

        # 3. Clean up noise and link the 'flexing' parts
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 4. Contour Filter: Find the rod by its verticality
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(mask)
        
        for cnt in contours:
            x, y, w_c, h_c = cv2.boundingRect(cnt)
            aspect_ratio = h_c / float(w_c)
            # Filter for tall, thin objects that are near the bottom
            if aspect_ratio > 2.5 and (y + h_c) > (h * 0.8):
                cv2.drawContours(final_mask, [cnt], -1, 255, -1)

        return final_mask

    def process_frame(self, frame):
        current_mask = self.get_rod_mask(frame)

        # 5. Temporal Smoothing (The 'Running Mask')
        if self.running_mask is None:
            self.running_mask = current_mask.astype(float)
        else:
            # Blend the current detection with previous history
            cv2.accumulateWeighted(current_mask, self.running_mask, 1.0 - self.alpha)

        # Threshold the blended mask to get a solid binary area for inpainting
        _, binary_mask = cv2.threshold(self.running_mask.astype(np.uint8), 50, 255, cv2.THRESH_BINARY)
        
        # 6. Dilate slightly to ensure we cover the 'glow' or edges of the rod
        dilate_kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.dilate(binary_mask, dilate_kernel, iterations=2)

        # 7. Inpaint: Fill the hole using surrounding textures
        # cv2.INPAINT_TELEA is generally faster for real-time video
        result = cv2.inpaint(frame, binary_mask, self.inpaint_radius, cv2.INPAINT_TELEA)
        
        return result, binary_mask


class DetectorBase:
    def __init__(self):
        # Overlay is (B,G,R)
        self.overlay = None

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

    def filter(self, frame):
        return frame


class Detector1(DetectorBase):
    def __init__(self):
        DetectorBase.__init__(self)

    def filter(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        lu, au, bu = cv2.split(lab)     # uint8
        aS = au.astype(np.int16) - 128
        bS = bu.astype(np.int16) - 128
        ab_diff = np.abs(aS - bS)
        ab_diff_output = ab_diff.astype(np.uint8)

        # L filter
        mask = cv2.inRange(lu, 128, 180)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        # a-b filter
        mask = cv2.inRange(ab_diff_output, 0, 4)
        result = cv2.bitwise_and(result, result, mask=mask)

        draw_line(ab_diff_output, 0, -1, (0, 255, 255), self.overlay)
        draw_line(lab, 0, -1, (0, 0, 255), self.overlay)

        return result


class Detector2(DetectorBase):
    def __init__(self):
        DetectorBase.__init__(self)

    def cv_opencv_optimized(self, sample):
        # Coefficient of Variation (CV)
        # Standard CV calculation using OpenCV's optimized core
        mu, sigma = cv2.meanStdDev(sample)
        mu_val = mu[0][0]
        sigma_val = sigma[0][0]
        return sigma_val / mu_val if mu_val > 0 else 0.0

    def get_cv_vectorized(self, strip):
        """
        Coefficient of Variation (CV)
        Calculates CV for all columns in a strip simultaneously.
        'strip' should be a (10, width) array.
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

    def filter_old(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        lu, au, bu = cv2.split(lab)     # uint8

        aS = au.astype(np.int16) - 128
        bS = bu.astype(np.int16) - 128
        ab_diff = np.abs(aS - bS)
        ab_diff = ab_diff.astype(np.uint8)

        # if False:
        #     # 2. Extract the bottom 10 rows for all columns
        #     # Shape will be (10, width)
        #     bottom_strip = lu[-10:, :]
        #     _, width = lu.shape
        #     cv_results = np.zeros(width, dtype=np.float32)

        #     for x in range(width):
        #         # Extract the 10-pixel vertical sample for this column
        #         column_sample = bottom_strip[:, x]
        #         cv_results[x] = self.cv_opencv_optimized(column_sample)
        # else:
        cv_lu = self.get_cv_vectorized(lu[-10:, :])
        cv_ab = self.get_cv_vectorized(ab_diff[-10:, :])

        cv_disp_lu_1d = np.clip(cv_lu * 1000, a_min=None, a_max=255)
        cv_disp_ab_1d = np.clip(cv_ab * 1000, a_min=None, a_max=255)
        draw_line(cv_disp_lu_1d, 0, -1, (0, 255, 255), self.overlay)
        draw_line(cv_disp_ab_1d, 0, -1, (255, 0, 255), self.overlay)

        return frame

    def filter(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        # lu, au, bu = cv2.split(lab)     # uint8
        lu = lab[:, :, 0]
        blur = cv2.GaussianBlur(lu, (21, 21), 0)  # kernel size

        cv_lu = self.get_cv_vectorized(lu[-10:, :])

        cv_disp_lu_1d = np.clip(cv_lu * 1000, a_min=None, a_max=255)
        draw_line(cv_disp_lu_1d, 0, -1, (0, 255, 255), self.overlay)

        rgb_result = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
        return rgb_result


class Main:
    def __init__(self):
        self.mx = 0
        self.my = 0
        self.zoom = 1
        self.view_org = False
        self.skip_num = 1

    def print_mouse_rgb(self, source, dest):
        x = self.mx
        y = self.my
        z = self.zoom
        b,g,r = source[y, x]
        text = f"X: {x}, Y: {y} | R: {r} G: {g} B: {b}"
        # cv2.rectangle(frame, (x + 10, y - 30), (x + 300, y), (0, 0, 0), -1)
        cv2.putText(dest, text,
            (10 * z, 20 * z),           # bottom-left coord
            cv2.FONT_HERSHEY_SIMPLEX,   # font
            z / 2,             # font scale
            (255, 255, 255),            # color
            z * 2)                  # line thickness

    def print_fps(self, loop_s, dest):
        fps = 1/loop_s if loop_s > 0 else 0
        ms = int(loop_s * 1000)
        text = f"{ms} ms, {fps:.2f} fps"
        z = self.zoom
        cv2.putText(dest, text,
            (10 * z, 30 * z),           # bottom-left coord
            cv2.FONT_HERSHEY_DUPLEX,    # font
            z,                          # font scale
            (0, 255, 255),              # color
            z )                         # line thickness

    def run(self):
        print("@@ Run")

        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TITLE, 1920//2, 1080//2)
        def _mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_MOUSEMOVE:
                self.mx = x
                self.my = y
        cv2.setMouseCallback(WINDOW_TITLE, _mouse_callback)

        width = 0
        height = 0
        loop_s = 0
        init_once = True
        frame_count = 0

        # detector = Detector1()
        detector = Detector2()

        cap = cv2.VideoCapture(VIDEOS[0])
        try:
            while cap.isOpened():
                start_loop_s = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                _skip_num = self.skip_num
                if _skip_num > 1:
                    if frame_count % _skip_num != 0:
                        continue

                if init_once:
                    height, width = frame.shape[:2]
                    print(f"Video size: {width}x{height}")
                    init_once = False

                detector.init_overlay(frame)
                result = detector.filter(frame)

                #self.print_mouse_rgb(frame, detector.overlay)
                self.print_fps(loop_s, detector.overlay)
                if self.view_org:
                    cv2.imshow(WINDOW_TITLE, detector.combine_overlay(frame))
                else:
                    cv2.imshow(WINDOW_TITLE, detector.combine_overlay(result))

                key = cv2.waitKey(FPS_MS) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('o'):
                    self.view_org = not self.view_org
                elif key == ord('s'):
                    self.skip_num = 1 + self.skip_num % 4
                elif key == ord('1'):
                    self.zoom = 1
                    cv2.resizeWindow(WINDOW_TITLE, width, height)
                elif key == ord('2'):
                    self.zoom = 2
                    cv2.resizeWindow(WINDOW_TITLE, width//2, height//2)
                elif key == ord('3'):
                    self.zoom = 3
                    cv2.resizeWindow(WINDOW_TITLE, width//3, height//3)
                elif key == ord('4'):
                    self.zoom = 4
                    cv2.resizeWindow(WINDOW_TITLE, width//4, height//4)
                end_loop_s = time.perf_counter()
                loop_s = end_loop_s - start_loop_s
        finally:
            cap.release()
            cv2.destroyAllWindows()

        print("@@ end")

if __name__ == "__main__":
    m = Main()
    m.run()

