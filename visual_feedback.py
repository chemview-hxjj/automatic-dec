# 氨基甲酸铵分解平衡常数测定控制程序的一部分，用于视觉判断
# 项目作者：李峙德，刘一弘
# 文件作者：刘一弘，李峙德
# 邮箱：contact@chemview.net
# 最后更新：2026-06-30
# Visual detection for the ammonium carbamate decomposition equilibrium constant experiment
# Project authors Li Zhide and Liu Yihong
# File authors Liu Yihong and Li Zhide
# Email contact@chemview.net
# Last updated 2026-06-30

import cv2
import numpy as np
from enum import Enum
from collections import deque
import matplotlib.pyplot as plt  # Used for standalone projection plots
import os                       # Used to create the data folder

class LiquidState(Enum):
    """Liquid level states with explicit left and right meanings"""
    LEFT_HIGH_LARGE_DIFF = 1     # Left level higher than right with a large gap
    LEFT_HIGH_SMALL_DIFF = 2     # Left level higher than right with a small gap
    RIGHT_HIGH_LARGE_DIFF = 3    # Right level higher than left with a large gap
    RIGHT_HIGH_SMALL_DIFF = 4    # Right level higher than left with a small gap
    RIGHT_TOO_LOW = 5            # Right level near the tube bottom or manual line
    LEFT_TOO_LOW = 6             # Left level near the tube bottom or manual line
    EQUILIBRIUM = 7              # Both levels are close enough
    RIGHT_NOTFOUND = 8           # No liquid column on the right side
    LEFT_NOTFOUND = 9            # No liquid column on the left side
    UNKNOWN = 0                  # No clear liquid column on either side

class UTubeDetector:
    def __init__(self, hsv_lower=[0, 100, 50], 
                 hsv_upper=[30, 255, 255], 
                 hsv_lower2=[120, 100, 50], 
                 hsv_upper2=[200, 255, 255], 
                 large_diff_threshold = 50, 
                 small_diff_threshold = 5, 
                 equilibrium_threshold = 5, 
                 bottom_ratio=0.1,          # Distance from the bottom line as a ratio
                 min_liquid_area=800,  
                 bubble_filter_kernel=(9,9),  
                 vertical_window=15,          
                 history_window=10,            
                 tube_width_ratio=0.1,
                 no_liquid_ratio=0.05):       
        """
        Set up the U tube detector with bottom line and manual line support
        bottom_ratio is the share above the bottom line treated as too low
        """
        # HSV bounds
        self.hsv_lower1 = np.array(hsv_lower)
        self.hsv_upper1 = np.array(hsv_upper)
        self.hsv_lower2 = np.array(hsv_lower2)
        self.hsv_upper2 = np.array(hsv_upper2)
        
        # Detection settings
        self.large_diff_threshold = large_diff_threshold
        self.small_diff_threshold = small_diff_threshold
        self.equilibrium_threshold = equilibrium_threshold
        self.bottom_ratio = bottom_ratio          # Too low ratio relative to the bottom line
        self.min_liquid_area = min_liquid_area
        self.vertical_window = vertical_window
        self.tube_width_ratio = tube_width_ratio
        self.no_liquid_ratio = no_liquid_ratio
        
        # Morphology kernels
        self.kernel = np.ones((7, 7), np.uint8)
        self.bubble_kernel = np.ones(bubble_filter_kernel, np.uint8)
        self.close_kernel = np.ones((11, 11), np.uint8)
        
        # State smoothing
        self.history_window = history_window
        self.left_level_history = deque(maxlen=history_window)
        self.right_level_history = deque(maxlen=history_window)
        self.state_history = deque(maxlen=history_window)
        
        # Bottom line state
        self.manual_bottom_line = None           # Manual bottom line y coordinate
        self.manual_bottom_set = False           # Whether a manual bottom line is active
        self.tube_bottom_line = None             # Detected U tube bottom line y coordinate
        self.drawing_mode = False                # Whether line drawing mode is active
        
        # Runtime state
        self.left_level = 0
        self.right_level = 0
        self.left_level_smoothed = 0
        self.right_level_smoothed = 0
        self.current_state = LiquidState.UNKNOWN
        self.left_tube_position = None  # Center x of the left tube in the frame
        self.right_tube_position = None # Center x of the right tube in the frame
        self.tube_width = None
        self.left_liquid_ratio = 0.0
        self.right_liquid_ratio = 0.0
        self.left_has_liquid = False    # Whether the left tube has a usable liquid column
        self.right_has_liquid = False   # Whether the right tube has a usable liquid column

        self.result = None
        self.cap = None
        self.frame = None
        self.display_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    def preprocess_image(self, image):
        """Preprocess the frame and tame the noise"""
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        return enhanced
    
    def detect_red_liquid(self, image):
        """Find red liquid regions and smooth out bubbles"""
        enhanced = self.preprocess_image(image)
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.hsv_lower1, self.hsv_upper1)
        mask2 = cv2.inRange(hsv, self.hsv_lower2, self.hsv_upper2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Keep the raw two range mask for export figure b
        self.si_mask_raw = red_mask.copy()
        
        # Morphology cleanup
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, self.close_kernel, iterations=2)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, self.bubble_kernel, iterations=1)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, self.kernel, iterations=1)
        
        # Keep the cleaned mask for export figure c
        self.si_mask_morph = red_mask.copy()
        
        # Find contours and filter small bubbles
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = []
        for contour in contours:
            if cv2.contourArea(contour) > self.min_liquid_area:
                filtered_contours.append(contour)
        if not filtered_contours:
            filtered_contours = contours
            
        return red_mask, filtered_contours
    
    def detect_tube_bottom(self, contours, image_height):
        """
        Detect the U tube bottom line from the lowest liquid contours
        Uses the lowest contour y position as the tube bottom
        """
        if not contours:
            self.tube_bottom_line = image_height  # Fall back to the image bottom when no contour is found
            return self.tube_bottom_line
        
        # Collect the lowest y coordinate from each contour
        bottom_ys = []
        for contour in contours:
            y, h = cv2.boundingRect(contour)[1], cv2.boundingRect(contour)[3]
            bottom_y = y + h  # Contour bottom y coordinate
            bottom_ys.append(bottom_y)
        
        # Add a small margin so the line is not glued to the contour
        self.tube_bottom_line = max(bottom_ys) + 10
        # Stay inside the frame
        self.tube_bottom_line = min(self.tube_bottom_line, image_height)
        return self.tube_bottom_line
    
    def identify_tube_positions(self, contours, image_width):
        """Find the left and right tube positions even when the contour is joined"""
        if not contours:
            return None, None
        
        # One connected contour means the two sides need to be split by position
        if len(contours) == 1:
            contour = contours[0]
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            self.tube_width = int(w * 0.3)  # Tube width is about thirty percent of the contour
            # Split the contour into left and right tube centers
            left_tube = max(0, min(image_width - 1, center_x - self.tube_width))
            right_tube = max(0, min(image_width - 1, center_x + self.tube_width))
            
            self.left_tube_position = left_tube
            self.right_tube_position = right_tube
            return left_tube, right_tube
        
        # Multiple contours are sorted so the smaller x value stays on the left
        tube_centers = []
        tube_widths = []
        for contour in contours:
            if len(contour) > 0:
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                tube_centers.append(center_x)
                tube_widths.append(w)
        
        if len(tube_centers) < 2:
            # Fall back to the single contour split
            contour = contours[0] if contours else None
            if contour is not None:
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                self.tube_width = int(w * 0.3)
                left_tube = max(0, min(image_width - 1, center_x - self.tube_width))
                right_tube = max(0, min(image_width - 1, center_x + self.tube_width))
                self.left_tube_position = left_tube
                self.right_tube_position = right_tube
                return left_tube, right_tube
            return None, None
        
        # Sort centers so the first is left and the last is right
        self.tube_width = int(np.mean(tube_widths) * 1.2)
        tube_centers.sort()  # Ascending x order
        self.left_tube_position = tube_centers[0]
        self.right_tube_position = tube_centers[-1]
        
        return self.left_tube_position, self.right_tube_position
    
    def calculate_roi_liquid_ratio(self, mask, x_center, width, image_height):
        """Measure liquid share and level inside one tube ROI"""
        # Limit the ROI to the frame
        x_start = max(0, x_center - width // 2)
        x_end = min(mask.shape[1], x_center + width // 2)
        roi = mask[:, x_start:x_end]
        roi_total_pixels = roi.shape[0] * roi.shape[1]
        
        # Count liquid pixels in the ROI
        liquid_pixels = np.sum(roi > 0)
        liquid_ratio = liquid_pixels / roi_total_pixels if roi_total_pixels > 0 else 0.0
        
        # Decide whether this looks like a real liquid column
        has_liquid = liquid_ratio >= self.no_liquid_ratio
        
        # Build the vertical projection
        if not has_liquid:
            return liquid_ratio, image_height, has_liquid
        
        # Each row stores the effective red pixel width
        vertical_proj = np.sum(roi, axis=1) / 255

        # Save projection data for export figure d
        self.si_vertical_proj = vertical_proj.copy()
        self.si_roi_width = width
        
        # Locate the meniscus by looking for the strongest projection jump
        # First differences show how quickly the liquid width changes downward
        # A wider blur keeps small burrs from winning
        smoothed_proj = cv2.GaussianBlur(vertical_proj.reshape(-1, 1), (1, 15), 0).flatten()
        gradients = np.diff(smoothed_proj)
        
        # The meniscus often shows up as a sharp positive jump
        # Search the upper part and avoid the bottom shrinkage
        search_limit = int(image_height * 0.85)
        if len(gradients) > 0:
            # Pick the row with the largest gradient
            max_grad_y = np.argmax(gradients[:search_limit])
            
            # Use it only if the width clears the noise floor
            if smoothed_proj[max_grad_y + 1] > width * 0.0001:
                detected_level = int(max_grad_y + 1)
                return liquid_ratio, detected_level, True
        
        # Fall back to the older rough threshold if the gradient search fails
        liquid_rows = [y for y in range(image_height) if vertical_proj[y] > width * 0.0001]
        if not liquid_rows:
            return liquid_ratio, image_height, False
        
        liquid_rows = sorted(liquid_rows)
        min_y = liquid_rows[0]
        
        return liquid_ratio, min_y, has_liquid
    
    def get_effective_bottom_line(self, image_height):
        """
        Return the bottom line currently used by the detector
        Manual line wins over detected tube bottom and image bottom
        """
        # Prefer the manually set line
        if self.manual_bottom_set and self.manual_bottom_line is not None:
            return self.manual_bottom_line
        # Then use the detected U tube bottom
        elif self.tube_bottom_line is not None:
            return self.tube_bottom_line
        # Last choice is the image bottom
        else:
            return image_height
    
    def judge_liquid_too_low(self, liquid_level, bottom_line):
        """
        Check whether the liquid level is close to the bottom line
        Larger y coordinates are lower in the image
        """
        # Threshold a fixed share above the bottom line
        too_low_threshold = bottom_line - (bottom_line * self.bottom_ratio)
        # Larger than the threshold means lower in the frame
        return liquid_level > too_low_threshold
    
    def find_liquid_levels(self, contours, image_shape):
        """Find the left and right liquid levels and update the bottom line"""
        height, width = image_shape[:2]
        left_level = height
        right_level = height
        
        # Reset the liquid column flags
        self.left_liquid_ratio = 0.0
        self.right_liquid_ratio = 0.0
        self.left_has_liquid = False
        self.right_has_liquid = False
        
        # Get the tube positions
        left_tube, right_tube = self.identify_tube_positions(contours, width)
        if left_tube is None or right_tube is None or self.tube_width is None:
            # Still refresh the bottom line when positions are missing
            self.detect_tube_bottom(contours, height)
            return left_level, right_level, left_level, right_level
        
        # Get the red mask
        red_mask, _ = self.detect_red_liquid(self.frame)
        
        # Update the U tube bottom line
        self.detect_tube_bottom(contours, height)
        
        # Analyze the left tube ROI
        self.left_liquid_ratio, left_level, self.left_has_liquid = self.calculate_roi_liquid_ratio(
            red_mask, left_tube, self.tube_width, height
        )
        # Analyze the right tube ROI
        self.right_liquid_ratio, right_level, self.right_has_liquid = self.calculate_roi_liquid_ratio(
            red_mask, right_tube, self.tube_width, height
        )
        
        # Smooth the level readings
        self.left_level_history.append(left_level)
        self.right_level_history.append(right_level)
        left_level_smoothed = int(np.mean(self.left_level_history))
        right_level_smoothed = int(np.mean(self.right_level_history))
        
        return left_level, right_level, left_level_smoothed, right_level_smoothed
    
    def analyze_liquid_state(self, left_levell_smoothed, right_levell_smoothed, image_height):
        """Classify the liquid state from level and column checks"""
        # Check missing columns first so left and right stay unambiguous
        if not self.left_has_liquid and not self.right_has_liquid:
            # No liquid column on either side
            current_state = LiquidState.UNKNOWN
        elif not self.left_has_liquid and self.right_has_liquid:
            # Left tube has no column and the liquid is only on the right
            current_state = LiquidState.LEFT_NOTFOUND
        elif not self.right_has_liquid and self.left_has_liquid:
            # Right tube has no column and the liquid is only on the left
            current_state = LiquidState.RIGHT_NOTFOUND
        # With both columns present check low level high side and balance
        else:
            # Get the active bottom line
            effective_bottom = self.get_effective_bottom_line(image_height)
            
            # Check whether either side is too close to the bottom
            left_too_low = self.judge_liquid_too_low(left_levell_smoothed, effective_bottom)
            right_too_low = self.judge_liquid_too_low(right_levell_smoothed, effective_bottom)
            
            if left_too_low:
                current_state = LiquidState.LEFT_TOO_LOW
            elif right_too_low:
                current_state = LiquidState.RIGHT_TOO_LOW
            else:
                # Compare height and balance
                height_diff = left_levell_smoothed - right_levell_smoothed
                abs_diff = abs(height_diff)
                if abs_diff <= self.equilibrium_threshold:
                    current_state = LiquidState.EQUILIBRIUM
                elif height_diff > 0:  # Right level y is smaller so the right side is higher
                    if abs_diff > self.large_diff_threshold:
                        current_state = LiquidState.RIGHT_HIGH_LARGE_DIFF
                    elif abs_diff > self.small_diff_threshold:
                        current_state = LiquidState.RIGHT_HIGH_SMALL_DIFF
                    else:
                        current_state = LiquidState.EQUILIBRIUM
                else:  # Left level y is smaller so the left side is higher
                    if abs_diff > self.large_diff_threshold:
                        current_state = LiquidState.LEFT_HIGH_LARGE_DIFF
                    elif abs_diff > self.small_diff_threshold:
                        current_state = LiquidState.LEFT_HIGH_SMALL_DIFF
                    else:
                        current_state = LiquidState.EQUILIBRIUM
        
        self.raw_state = current_state
        
        # Smooth the state changes
        self.state_history.append(current_state)
        if len(self.state_history) >= self.history_window:
            state_counts = {}
            for state in self.state_history:
                state_counts[state] = state_counts.get(state, 0) + 1
            current_state = max(state_counts, key=state_counts.get)
        
        return current_state
    
    def set_manual_bottom_line(self, y_coord, image_height):
        """Set the manual bottom line from outside the detector"""
        # Clamp inside the frame
        self.manual_bottom_line = max(0, min(image_height, y_coord))
        self.manual_bottom_set = True
        print(f"已设置人工底部线:y={self.manual_bottom_line} 像素")
    
    def clear_manual_bottom_line(self):
        """Clear the manual bottom line"""
        self.manual_bottom_line = None
        self.manual_bottom_set = False
        print("已清除人工底部线,恢复自动检测U型管底部")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback used to draw the bottom line"""
        if self.drawing_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                # Left click sets the bottom line
                self.set_manual_bottom_line(y, param['image_height'])
                self.drawing_mode = False
            elif event == cv2.EVENT_RBUTTONDOWN:
                # Right click clears the bottom line
                self.clear_manual_bottom_line()
                self.drawing_mode = False
    
    def get_liquid_state(self):
        """Read the current liquid state"""
        image = self.frame
        red_mask, contours = self.detect_red_liquid(image)
        self.left_level, self.right_level,self.left_level_smoothed,self.right_level_smoothed = self.find_liquid_levels(contours, image.shape)
        self.current_state = self.analyze_liquid_state(
            self.left_level, self.right_level, image.shape[0]
        )
        
        # Calculate the height difference
        height_diff = self.left_level_smoothed - self.right_level_smoothed
        usm_height_diff = self.left_level - self.right_level
        # Include bottom line details in the result
        effective_bottom = self.get_effective_bottom_line(image.shape[0])
        self.result = {
            'state': self.current_state,
            'state_code': self.current_state.value,
            'raw_state_code':self.raw_state.value,
            'state_description': self.get_state_description(self.current_state),
            'left_level': self.left_level_smoothed,
            'right_level': self.right_level_smoothed,
            'usm_left_level': self.left_level,
            'usm_right_level':self.right_level,
            'usm_height_difference':abs(usm_height_diff),
            'height_difference': abs(height_diff),
            'height_diff_raw': height_diff,
            'left_tube_position': self.left_tube_position,
            'right_tube_position': self.right_tube_position,
            'left_liquid_ratio': round(self.left_liquid_ratio, 3),
            'right_liquid_ratio': round(self.right_liquid_ratio, 3),
            'left_has_liquid': self.left_has_liquid,
            'right_has_liquid': self.right_has_liquid,
            'tube_bottom_line': self.tube_bottom_line,          # Detected tube bottom line
            'manual_bottom_line': self.manual_bottom_line,      # Manual bottom line
            'manual_bottom_set': self.manual_bottom_set,        # Whether the manual line is active
            'effective_bottom_line': effective_bottom,           # Bottom line currently used
            'image_height': image.shape[0],
            'image_width': image.shape[1],
            'mask': red_mask,
            'contours': contours
        }

        self.display_frame = self.draw_detection_result(image, self.result)
    
    def get_state_description(self, state):
        """Return the display text for a liquid state"""
        descriptions = {
            LiquidState.LEFT_HIGH_LARGE_DIFF: "LEFT HIGH - LARGE DIFF",
            LiquidState.LEFT_HIGH_SMALL_DIFF: "LEFT HIGH - SMALL DIFF", 
            LiquidState.RIGHT_HIGH_LARGE_DIFF: "RIGHT HIGH - LARGE DIFF",
            LiquidState.RIGHT_HIGH_SMALL_DIFF: "RIGHT HIGH - SMALL DIFF",
            LiquidState.LEFT_TOO_LOW: "LEFT TOO LOW (NEAR BOTTOM LINE)",
            LiquidState.RIGHT_TOO_LOW: "RIGHT TOO LOW (NEAR BOTTOM LINE)",
            LiquidState.EQUILIBRIUM: "EQUILIBRIUM",
            LiquidState.RIGHT_NOTFOUND: "RIGHT NOTFOUND (LIQUID ONLY IN LEFT TUBE)",
            LiquidState.LEFT_NOTFOUND: "LEFT NOTFOUND (LIQUID ONLY IN RIGHT TUBE)",
            LiquidState.UNKNOWN: "UNKNOWN (BOTH TUBES HAVE NO LIQUID)"
        }
        return descriptions.get(state, "UNKNOWN")
    
    def draw_detection_result(self, image, result):
        """Draw detection overlays on the frame"""
        height, width = image.shape[:2]
        output_image = image.copy()
        
        # Draw contours
        cv2.drawContours(output_image, result['contours'], -1, (0, 100, 0), 2)
        
        # Draw tube ROIs red when empty and blue when filled
        if self.left_tube_position and self.tube_width:
            x1 = max(0, self.left_tube_position - self.tube_width // 2)
            x2 = min(width, self.left_tube_position + self.tube_width // 2)
            roi_color = (0, 0, 150) if not self.left_has_liquid else (150, 0, 0)
            cv2.rectangle(output_image, (x1, 0), (x2, height), roi_color, 2)
            # Label the left ROI liquid share
            ratio_text = f"L:{result['left_liquid_ratio']*100:.1f}%"
            cv2.putText(output_image, ratio_text, (x1 - 60, 450), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 2)
        
        if self.right_tube_position and self.tube_width:
            x1 = max(0, self.right_tube_position - self.tube_width // 2)
            x2 = min(width, self.right_tube_position + self.tube_width // 2)
            roi_color = (0, 0, 150) if not self.right_has_liquid else (150, 0, 0)
            cv2.rectangle(output_image, (x1, 0), (x2, height), roi_color, 2)
            # Label the right ROI liquid share
            ratio_text = f"R:{result['right_liquid_ratio']*100:.1f}%"
            cv2.putText(output_image, ratio_text, (x1 + 40, 450), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 2)
        
        # Draw bottom line overlays
        # Detected U tube bottom line in green
        if result['tube_bottom_line'] is not None:
            cv2.line(output_image, (0, result['tube_bottom_line']), 
                    (width, result['tube_bottom_line']), (0, 100, 0), 2)
            cv2.putText(output_image, f"Tube Bottom: {result['tube_bottom_line']}", 
                       (10, result['tube_bottom_line']-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 0), 2)
        
        # Manual bottom line in red
        if result['manual_bottom_set'] and result['manual_bottom_line'] is not None:
            cv2.line(output_image, (0, result['manual_bottom_line']), 
                    (width, result['manual_bottom_line']), (0, 0, 150), 3)
            cv2.putText(output_image, f"Manual Bottom: {result['manual_bottom_line']}", 
                       (10, result['manual_bottom_line']+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 150), 2)
        
        # Too low threshold based on the active bottom line
        effective_bottom = result['effective_bottom_line']
        too_low_threshold = effective_bottom - (effective_bottom * self.bottom_ratio)
        cv2.line(output_image, (0, int(too_low_threshold)), 
                (width, int(too_low_threshold)), (150, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(output_image, f"Too Low Threshold: {int(too_low_threshold)}", 
                   (width-250, int(too_low_threshold)+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 0, 0), 2)
        
        # Draw liquid level lines
        
        if result['left_level'] < height:
            cv2.line(output_image, (0, result['left_level']), 
                    (width, result['left_level']), (0, 0, 150), 3)
            cv2.putText(output_image, f"Left Level", (10, result['left_level']-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 150), 2)
        
        if result['right_level'] < height:
            cv2.line(output_image, (0, result['right_level']), 
                    (width, result['right_level']), (0, 100, 0), 3)
            cv2.putText(output_image, f"Right Level", (10, result['right_level']-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 0), 2)
        
        # Mark tube positions
        if result['left_tube_position'] is not None:
            cv2.line(output_image, (result['left_tube_position'], 0), 
                    (result['left_tube_position'], height), (150, 0, 0), 1)
            cv2.putText(output_image, "LEFT", (result['left_tube_position']-60, 465),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 0, 0), 2)
        if result['right_tube_position'] is not None:
            cv2.line(output_image, (result['right_tube_position'], 0), 
                    (result['right_tube_position'], height), (150, 0, 0), 1)
            cv2.putText(output_image, "RIGHT", (result['right_tube_position']+40, 465),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 0, 0), 2)
        
        # Show state text
        state_text = self.get_state_description(result['state'])
        text_color = (0, 0, 150) if result['state_code'] in [8, 9] else (0, 0, 0)
        cv2.putText(output_image, state_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        
        # Show level readings and difference
        level_text = f"Left: {result['left_level']}, Right: {result['right_level']}"
        cv2.putText(output_image, level_text, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        diff_text = f"Difference: {result['height_difference']}"
        cv2.putText(output_image, diff_text, (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Show bottom line info
        bottom_text = f"Effective Bottom: {result['effective_bottom_line']} | Ratio: {self.bottom_ratio*100:.1f}%"
        cv2.putText(output_image, bottom_text, (10, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 0, 0), 2)
        
        # Show the drawing mode hint
        if self.drawing_mode:
            cv2.putText(output_image, "DRAW MODE: LEFT-CLICK TO SET BOTTOM | RIGHT-CLICK TO CLEAR", 
                       (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 150), 2)
        
        return output_image
    
    def set_thresholds(self, large_diff=None, small_diff=None, equilibrium=None, 
                       bottom=None, min_area=None, vertical_window=None, history_window=None,
                       no_liquid_ratio=None, bottom_ratio=None):
        """Update detection thresholds including the bottom ratio"""
        if large_diff is not None:
            self.large_diff_threshold = large_diff
        if small_diff is not None:
            self.small_diff_threshold = small_diff
        if equilibrium is not None:
            self.equilibrium_threshold = equilibrium
        if bottom is not None:
            self.bottom_threshold = bottom
        if min_area is not None:
            self.min_liquid_area = min_area
        if vertical_window is not None:
            self.vertical_window = vertical_window
        if history_window is not None:
            self.history_window = history_window
            self.left_level_history = deque(maxlen=history_window)
            self.right_level_history = deque(maxlen=history_window)
            self.state_history = deque(maxlen=history_window)
        if no_liquid_ratio is not None:
            self.no_liquid_ratio = no_liquid_ratio
        if bottom_ratio is not None:
            self.bottom_ratio = bottom_ratio  # Update the too low ratio
            
    def save_each_si_figure(self, folder="paper_data"):
        """
        Export clean step by step images and plot source data for personal data collection
        """
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        if self.frame is None:
            print("\n[错误] 没有捕获到有效的图像帧，无法导出数据。")
            return

        # Export figure a as the clean raw camera frame
        cv2.imwrite(os.path.join(folder, "fig_a_original.jpg"), self.frame)

        # Export figure b as the raw two range mask
        if hasattr(self, 'si_mask_raw'):
            cv2.imwrite(os.path.join(folder, "fig_b_raw_mask.png"), self.si_mask_raw)

        # Export figure c as the cleaned morphology mask
        if hasattr(self, 'si_mask_morph'):
            cv2.imwrite(os.path.join(folder, "fig_c_morph_mask.png"), self.si_mask_morph)

        # Export figure d as a standalone vertical projection plot and CSV
        if hasattr(self, 'si_vertical_proj'):
            # Render and save the plot
            fig, ax = plt.subplots(figsize=(6, 5))
            y_indices = np.arange(len(self.si_vertical_proj))
            
            # Plot the vertical liquid pixel density
            ax.plot(self.si_vertical_proj, y_indices, color='crimson', linewidth=2.5, label='Liquid Pixel Count')
            
            # Draw the level threshold line
            thresh_line = self.si_roi_width * 0.1
            ax.axvline(x=thresh_line, color='blue', linestyle=':', linewidth=1.5, label='Noise Threshold')
            
            # Mark the detected level using the smoothed right side value
            ax.axhline(y=self.right_level_smoothed, color='darkgreen', linestyle='--', linewidth=1.5, 
                       label=f'Detected Level (y={self.right_level_smoothed})')
            
            ax.set_xlabel("Effective Liquid Pixel Width (pixels)", fontsize=11)
            ax.set_ylabel("Image Y-Coordinate (pixels)", fontsize=11)
            ax.set_ylim(0, len(self.si_vertical_proj))
            ax.invert_yaxis()  # Match the plot y direction to the camera frame
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(loc='lower right', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(os.path.join(folder, "fig_d_projection.png"), dpi=300) # 300 DPI for publication style output
            plt.close()

            # Export the plot source data as CSV
            import csv
            csv_filename = os.path.join(folder, "fig_d_projection_data.csv")
            try:
                with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    # Write the header
                    writer.writerow(["Y-Coordinate(pixels)", "Effective_Liquid_Pixel_Width(pixels)"])
                    # Write y coordinates and matching effective liquid width
                    for y, width in zip(y_indices, self.si_vertical_proj):
                        writer.writerow([y, round(float(width), 3)])
                    writer.writerow(["Threshhold_Line",thresh_line])
                    writer.writerow(["Right_Smoothed_Level",self.right_level_smoothed])
            except Exception as e:
                print(f"\n[错误] 保存CSV源数据失败: {e}")
            
        print(f"\n[数据收集成功] 4张独立的论文无水印图及1份CSV数据已保存至: '{folder}/' 文件夹。")
# Shared detector instance
_detector_instance = None

def get_utube_detector(hsv_lower=[0, 120, 70], hsv_upper=[10, 255, 255], 
                      hsv_lower2=[170, 120, 70], hsv_upper2=[180, 255, 255]):
    """Return the shared U tube detector"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = UTubeDetector(
            hsv_lower=hsv_lower, 
            hsv_upper=hsv_upper,
            hsv_lower2=hsv_lower2,
            hsv_upper2=hsv_upper2,
            min_liquid_area=300,      
            bubble_filter_kernel=(9,9),
            vertical_window=10,       
            history_window=8,         
            no_liquid_ratio=0.05,
            bottom_ratio=0.1          # Too low ratio
        )
    return _detector_instance

def analyze_liquid_state(image, detector=None):
    """Shared entry point for liquid state analysis"""
    if detector is None:
        detector = get_utube_detector()
    detector.frame = image
    detector.get_liquid_state()
    return detector.result, detector.display_frame

# Live camera test with manual bottom line interaction and no mirroring
def main():
    detector = get_utube_detector()
    detector.set_thresholds(
        min_area=300,          
        vertical_window=10,    
        history_window=8,
        no_liquid_ratio=0.05,
        bottom_ratio=0.1       # Treat the lower ten percent near the bottom line as too low
    )
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    # Create the window and bind the mouse callback
    cv2.namedWindow('U-Tube Liquid Detection (Bottom Line)')
    def mouse_cb(event, x, y, flags, param):
        detector.mouse_callback(event, x, y, flags, {'image_height': param['height']})
    cv2.setMouseCallback('U-Tube Liquid Detection (Bottom Line)', mouse_cb, 
                        {'height': 480})  # Pass the image height
    
    print("="*80)
    print("U型管液面检测程序(底部线判定+人工划线版)")
    print("按键说明：")
    print("  q - 退出程序 | s - 保存图像 | t - 调整平衡阈值")
    print("  a - 调整最小液体面积 | v - 调整垂直窗口 | h - 调整平滑窗口")
    print("  r - 调整无液柱阈值 | b - 进入/退出划线模式（设置底部线）")
    print("  k - 调整过低判定比例(如0.1=10%) | c - 清除人工底部线")
    print("="*80)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("错误：无法读取帧")
            break
        # Keep the camera frame unmirrored
        
        result, display_frame = analyze_liquid_state(frame, detector)
        
        cv2.imshow('U-Tube Liquid Detection (Bottom Line)', display_frame)
        cv2.imshow('Red Mask (Filtered)', result['mask'])
        
        # Print detailed state with bottom line info
        state_info = (
            f"状态: {result['state_description']} | 左占比: {result['left_liquid_ratio']*100:.1f}% | "
            f"右占比: {result['right_liquid_ratio']*100:.1f}% | 有效底部线: {result['effective_bottom_line']} | "
            f"人工线: {result['manual_bottom_set']} | 过低比例: {detector.bottom_ratio*100:.1f}%"
        )
        print(state_info, end='\r')
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('utube_capture_bottom_line.jpg', frame)
            cv2.imwrite('utube_detection_result_bottom_line.jpg', display_frame)
            cv2.imwrite('utube_mask_bottom_line.jpg', result['mask'])
            print("\n图像已保存:capture/detection_result/mask_bottom_line.jpg")
        elif key == ord('i'):  
            detector.save_each_si_figure(folder="paper_data")
        elif key == ord('t'):
            try:
                new_val = int(input(f"\n当前平衡阈值: {detector.equilibrium_threshold}\n输入新值: "))
                detector.set_thresholds(equilibrium=new_val)
                print(f"平衡阈值更新为: {detector.equilibrium_threshold}")
            except ValueError:
                print("输入错误！")
        elif key == ord('a'):
            try:
                new_val = int(input(f"\n当前最小液体面积: {detector.min_liquid_area}\n输入新值: "))
                detector.set_thresholds(min_area=new_val)
                print(f"最小液体面积更新为: {detector.min_liquid_area}")
            except ValueError:
                print("输入错误！")
        elif key == ord('v'):
            try:
                new_val = int(input(f"\n当前垂直窗口: {detector.vertical_window}\n输入新值: "))
                detector.set_thresholds(vertical_window=new_val)
                print(f"垂直窗口更新为: {detector.vertical_window}")
            except ValueError:
                print("输入错误！")
        elif key == ord('h'):
            try:
                new_val = int(input(f"\n当前平滑窗口: {detector.history_window}\n输入新值: "))
                detector.set_thresholds(history_window=new_val)
                print(f"平滑窗口更新为: {detector.history_window}")
            except ValueError:
                print("输入错误！")
        elif key == ord('r'):
            try:
                new_val = float(input(f"\n当前无液柱阈值: {detector.no_liquid_ratio}\n输入新值（如0.05）: "))
                detector.set_thresholds(no_liquid_ratio=new_val)
                print(f"无液柱阈值更新为: {detector.no_liquid_ratio}")
            except ValueError:
                print("输入错误！")
        elif key == ord('b'):
            # Toggle drawing mode
            detector.drawing_mode = not detector.drawing_mode
            mode_text = "开启" if detector.drawing_mode else "关闭"
            print(f"\n划线模式{mode_text}：左键点击设置底部线，右键点击清除")
        elif key == ord('k'):
            # Adjust the too low ratio
            try:
                new_val = float(input(f"\n当前过低判定比例: {detector.bottom_ratio}\n输入新值(如0.1=10%): "))
                detector.set_thresholds(bottom_ratio=new_val)
                print(f"过低判定比例更新为: {detector.bottom_ratio}")
            except ValueError:
                print("输入错误！")
        elif key == ord('c'):
            # Clear the manual bottom line
            detector.clear_manual_bottom_line()
            print("\n已清除人工底部线,恢复自动检测U型管底部")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n程序已退出")

# Single image test helper
def test_with_image(image_path):
    detector = get_utube_detector()
    detector.set_thresholds(
        min_area=300,       
        vertical_window=10,
        history_window=3,
        no_liquid_ratio=0.05,
        bottom_ratio=0.1
    )
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法读取图像 {image_path}")
        return
    
    # Bind the mouse callback for single image testing
    cv2.namedWindow('U-Tube Detection Result (Bottom Line)')
    def mouse_cb(event, x, y, flags, param):
        detector.mouse_callback(event, x, y, flags, {'image_height': image.shape[0]})
    cv2.setMouseCallback('U-Tube Detection Result (Bottom Line)', mouse_cb)
    
    result, display_image = analyze_liquid_state(image, detector)
    
    cv2.imshow('U-Tube Detection Result (Bottom Line)', display_image)
    cv2.imshow('Red Mask (Filtered)', result['mask'])
    
    print("\n" + "="*80)
    print("图像检测结果（底部线判定+人工划线版）")
    print("="*80)
    print(f"图像尺寸: {image.shape[1]}x{image.shape[0]}")
    print(f"状态码: {result['state_code']}")
    print(f"状态描述: {result['state_description']}")
    print(f"左液面高度: {result['left_level']} 像素")
    print(f"右液面高度: {result['right_level']} 像素")
    print(f"液面高度差: {result['height_difference']} 像素")
    print(f"左ROI液体占比: {result['left_liquid_ratio']*100:.1f}%")
    print(f"右ROI液体占比: {result['right_liquid_ratio']*100:.1f}%")
    print(f"U型管底部线: {result['tube_bottom_line']} 像素")
    print(f"人工底部线: {result['manual_bottom_line']} 像素 (设置: {result['manual_bottom_set']})")
    print(f"有效底部线: {result['effective_bottom_line']} 像素")
    print(f"过低判定比例: {detector.bottom_ratio*100:.1f}%")
    print(f"左管过低阈值: {result['effective_bottom_line'] - (result['effective_bottom_line']*detector.bottom_ratio)} 像素")
    print(f"右管过低阈值: {result['effective_bottom_line'] - (result['effective_bottom_line']*detector.bottom_ratio)} 像素")
    print(f"左管是否过低: {detector.judge_liquid_too_low(result['left_level'], result['effective_bottom_line'])}")
    print(f"右管是否过低: {detector.judge_liquid_too_low(result['right_level'], result['effective_bottom_line'])}")
    print(f"左管是否有液柱: {result['left_has_liquid']}")
    print(f"右管是否有液柱: {result['right_has_liquid']}")
    print("="*80)
    print("提示:按b键进入划线模式,左键点击设置底部线,右键清除")
    print("="*80)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            detector.drawing_mode = not detector.drawing_mode
            mode_text = "开启" if detector.drawing_mode else "关闭"
            print(f"\n划线模式{mode_text}：左键点击设置底部线，右键点击清除")
        elif key == ord('c'):
            detector.clear_manual_bottom_line()
            print("\n已清除人工底部线")
            # Analyze again after clearing the line
            result, display_image = analyze_liquid_state(image, detector)
            cv2.imshow('U-Tube Detection Result (Bottom Line)', display_image)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # test_with_image("test_image.jpg")  # Single image test
    main()  # Live detection
