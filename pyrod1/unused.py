#!/usr/bin/python


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


class Detector1(DetectorBase):
    def __init__(self):
        super().__init__()

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

    def cv_opencv_optimized_unused(self, sample):
        # Coefficient of Variation (CV)
        # Standard CV calculation using OpenCV's optimized core
        mu, sigma = cv2.meanStdDev(sample)
        mu_val = mu[0][0]
        sigma_val = sigma[0][0]
        return sigma_val / mu_val if mu_val > 0 else 0.0

    def detect_rod_prominence_unused(self, cv_vector, center_weight=2.5):
        """
        Finds the rod by looking for the most prominent 'valley'
        in the Coefficient of Variation signal.
        """
        # 1. Invert the signal: Low CV (rod) becomes a high peak
        inverted_cv = -cv_vector

        # 2. Find peaks based on Prominence
        # We don't use a hard 'height' threshold; we let prominence do the work.
        # width=(10, 50) allows for some blurring/flexing around the 30px target.
        peaks, props = scipy.signal.find_peaks(
            inverted_cv,
            prominence=0.10, # Minimum 'depth' of the valley to be considered
            width=self.rod_delta_px,    # Looking for our ~30px rod
            rel_height=0.5     # Calculate width at 50% of the prominence
        )

        if len(peaks) == 0:
            return None

        img_center = len(cv_vector) / 2
        best_candidate = None
        max_score = -1.0

        scores = []
        for i in range(len(peaks)):
            idx = peaks[i]
            prom = props['prominences'][i]
            width = props['widths'][i]

            # 3. Scoring: Combine Prominence and Center-Weighting
            # Distance penalty (0.0 at center, increasing to ~0.7 at edges)
            dist_factor = 1.0 / (1.0 + (abs(idx - img_center) / img_center) * center_weight)

            # Total score: How 'valley-like' it is * How centered it is
            score = prom * dist_factor

            left_px = int(props['left_ips'][i])
            right_px = int(props['right_ips'][i])

            # Draw segment for debug
            y = self.height - GRAPH_Y_OFFSET - min(255, int(100 * score))
            cv2.line(self.overlay, (left_px, y), (right_px, y), (0, 0, 255), 2)
            scores.append(score)

            if score > max_score:
                max_score = score
                best_candidate = {
                    'center': idx,
                    'width': width,
                    'prominence': prom,
                    'score': score,
                    'boundaries': (left_px, right_px)
                }

            print(scores)
        return best_candidate

    def draw_peaks(self, peaks, threshold_y, color_threshold, color_peaks, dest):
        y = self.height - int(threshold_y) - GRAPH_Y_OFFSET

        w2 = self.rod_width_px // 2

        cv2.line(dest, (0, y), (self.width, y), color_threshold, 1)
        y -= 4

        for p in peaks:
            cv2.line(dest, (p - w2, y), (p + w2, y), color_peaks, 2)
            y -= 2

    def find_rod_valleys_unused(self, cv_under_threshold):
        """
        Finds the rod by searching for low-variance valleys in a 1D signal.
        """
        if self.current_rod is None:
            score_center = cv_under_threshold.size / 2
            rod_left = -1
            rod_right = -1
        else:
            score_center = self.current_rod.center()
            rod_left = self.current_rod.left
            rod_right = self.current_rod.right

        # Group contiguous 'valley' pixels
        # labels is an array where each valley is numbered 1, 2, 3...
        labels, num_features = scipy.ndimage.label(cv_under_threshold)

        candidates = []

        y = self.height - GRAPH_Y_OFFSET

        rod_width = self.rod_width_px
        min_width = self.rod_w_range_px[0]
        max_width = self.rod_w_range_px[1]


        for i in range(1, num_features + 1):
            indices = np.where(labels == i)[0]
            width = len(indices)
            left_px = indices[0].item()
            right_px = indices[-1].item()
            midpoint = (left_px + right_px) / 2

            # Apply Width Constraints
            # yet check any segment overlapping the current rod
            cond_width = min_width <= width <= max_width
            cond_middle = rod_left <= midpoint <= rod_right
            if cond_width or cond_middle:

                # Calculate Center Score
                # Lower distance to center = smaller score -- we want the lowest score
                delta_center = midpoint - score_center
                score = abs(delta_center)
                # Score is also degraded by how much width differs from expected width
                score += abs(width - rod_width) / 10

                if cond_middle:
                    # We want to mostly use the old rod position, slightly shifted towards
                    # the new midpoint; we're trying to keep the same width.
                    left_px = int(self.weight(rod_left, rod_left + delta_center))
                    right_px = int(self.weight(rod_right, rod_right + delta_center))

                candidates.append( Rod(left_px, right_px, score) )

                ys = int(y - min(score / 2, 255))
                cv2.line(self.overlay, (left_px, ys), (right_px, ys), (255, 0, 0), 3)

        # Find best match (lowest score)
        if not candidates:
            return None
        # print("@@ ", self.last_threshold, " >> ", candidates)
        best_candidate = min(candidates, key=lambda x: x.score)

        # Ignore the best candidate if its score is drastically worse than the current one.
        # Since the score is a number of pixels off the center of the current rod, we can
        # compare the score delta to the rod width.
        if self.current_rod is not None:
            curr_score = self.current_rod.score
            if best_candidate.score > curr_score + 2 * rod_width:
                return None

        return best_candidate

    def merge_rod(self, new_rod):
        if new_rod is None:
            return
        if self.current_rod is None:
            self.current_rod = new_rod
        else:
            old = self.current_rod
            new_center = new_rod.center()
            old_center = old.center()

            # # Ignore new rod if it has moved by more than N rod widths
            # # For testing: we trigger a pause
            # delta_center = abs(new_center - old_center)
            # delta_threshold = 3 * self.rod_width_px
            # if delta_center > delta_threshold:
            #     self.trigger_pause = True

            new_rod = Rod(
                left=self.weight(old.left, new_rod.left, 0.5),
                right=self.weight(old.right, new_rod.right, 0.5),
                score=self.weight(old.score, new_rod.score, 0.5)
            )
            self.current_rod = new_rod
            # print("@@ new rod:", new_rod)
            # print("@@ delta", delta_center, "<", delta_threshold, " @@ ", old, " >>> ", self.current_rod)
            # else:
            #     print("@@ delta", delta_center, ">=", delta_threshold)
        return self.current_rod



if __name__ == "__main__":
    raise("unused stuff")

