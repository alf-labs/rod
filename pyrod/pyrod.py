#!/usr/bin/python
# Ensure we're running from the Virtual Env version
import os
if not "VIRTUAL_ENV" in os.environ:
    print("ERROR: Run this from venv using 'source ./venv_catd/bin/activate' first")
    exit(1)
IS_RPI = os.path.isfile("/etc/rpi-issue")

import base64
import sys

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


def detect_by_color_lab(frame):
    # image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    channel = image[:, :, 0] # L

    mask = cv2.inRange(channel, 120, 150)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

# lower_gray = np.array([125, 125, 125])
# upper_gray = np.array([131, 131, 131])
# mask = cv2.inRange(image, lower_gray, upper_gray)
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



class Main:
    def __init__(self):
        pass

    def run(self):
        print("@@ Run")

        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TITLE, 1920//2, 1080//2)

        remover = RodRemover(alpha=0.7) # Adjust alpha for more/less 'memory'

        cap = cv2.VideoCapture(VIDEOS[0])
        height = 0
        width = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # result = detect_edges_sobel_canny(frame)
                result = detect_by_color_lab(frame)
                # result = detect_by_color_hsv(frame)
                # result = detect_hough_lines(frame)

                # clean_frame, mask_viz = remover.process_frame(frame)
                # result = mask_viz

                cv2.imshow(WINDOW_TITLE, result)

                if width == 0:
                    height, width = frame.shape[:2]

                key = cv2.waitKey(FPS_MS) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('1'):
                    cv2.resizeWindow(WINDOW_TITLE, width, height)
                elif key == ord('2'):
                    cv2.resizeWindow(WINDOW_TITLE, width//2, height//2)
                elif key == ord('3'):
                    cv2.resizeWindow(WINDOW_TITLE, width//3, height//3)
                elif key == ord('4'):
                    cv2.resizeWindow(WINDOW_TITLE, width//4, height//4)

        finally:
            cap.release()
            cv2.destroyAllWindows()

        print("@@ end")

if __name__ == "__main__":
    m = Main()
    m.run()

