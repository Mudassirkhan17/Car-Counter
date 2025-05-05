import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import time
import pandas as pd
from datetime import datetime
import os  # for file existence checks
try:
    import openpyxl
except ImportError:
    print("Warning: openpyxl not installed, Excel segment sheets may not be supported. Falling back to CSV.")

cap = cv2.VideoCapture("location1.MTS")  # For Video

model = YOLO("yolov8n.pt")
model.to('cpu')

# Complete COCO classes list
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Define vehicle classes indices for COCO dataset
vehicle_classes = set([1, 2, 3, 5, 7])  # bicycle(1), car(2), motorcycle(3), bus(5), truck(7)

# Define minimum size thresholds for vehicle types
MIN_TRUCK_HEIGHT = 170
MIN_BUS_HEIGHT = 190

# Minimum detection size (objects smaller than this will be ignored)
MIN_DETECTION_SIZE = 48  # pixels - objects smaller than 15x15 are too small to classify reliably

# Confidence thresholds for each vehicle class
CAR_CONFIDENCE_THRESHOLD = 0.48
TRUCK_CONFIDENCE_THRESHOLD = 0.85
BUS_CONFIDENCE_THRESHOLD = 0.85
MOTORBIKE_CONFIDENCE_THRESHOLD = 0.48  # Highest threshold to prevent car misclassification
HIGH_CONFIDENCE_THRESHOLD = 0.75  # Threshold for high-confidence detections for small objects

# Load graphics image once
imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
# Commented out noisy print statements
# print("Warning: Could not load graphics.png, continuing without overlay")
if imgGraphics is None:
    imgGraphics = None

# Initialize tracker with optimized parameters
tracker = Sort(max_age=30, min_hits=1, iou_threshold=0.45)

# Load mask image as ROI filter
mask_img = cv2.imread("laneB(1).png", cv2.IMREAD_GRAYSCALE)
# Commented out noisy print statements
# print("Warning: Could not load mask file ml2.png, proceeding without mask")
if mask_img is None:
    mask_3ch = None
else:
    _, mask_bin = cv2.threshold(mask_img, 128, 255, cv2.THRESH_BINARY)
    mask_3ch = cv2.merge([mask_bin, mask_bin, mask_bin])

# Define counting lines
# limitsUp = [614, 396, 1038, 387]
# y axis making it more negative mean putting it up
limitsUp = [120-15, 390-30, 610-15, 389-30]
# limitsDown = [514+100, 396+40, 1038+100, 387+40]
limitsDown = [80-15, 401+20, 610-15, 399+30]

# Pre-calculate boundary ranges
limitsUp_y_min = limitsUp[1] - 57
limitsUp_y_max = limitsUp[1] + 57
limitsDown_y_min = limitsDown[1] - 57 
limitsDown_y_max = limitsDown[1] + 57

# Wider boundary for bikes (they're smaller and can be missed in narrow boundaries)
bike_limitsUp_y_min = limitsUp[1] - 50
bike_limitsUp_y_max = limitsUp[1] + 50
bike_limitsDown_y_min = limitsDown[1] - 50
bike_limitsDown_y_max = limitsDown[1] + 50

# Pre-calculate line coordinates for drawing
line_coords = {
    'up': ((limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3])),
    'down': ((limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]))
}

# Initialize sets for counting
carCountUp = set()
truckCountUp = set()
busCountUp = set()
motorbikeCountUp = set()

carCountDown = set()
truckCountDown = set()
busCountDown = set()
motorbikeCountDown = set()

# Tracking dictionaries
vehicleTypes = {}
vehicle_positions = {}
vehicle_first_types = {}

# Enhanced tracking to maintain identity despite ID changes
class EnhancedObjectTracker:
    def __init__(self, max_history=30):
        self.tracked_objects = {}  # Dict mapping stable IDs to current tracker IDs
        self.object_history = {}   # Dict mapping stable IDs to position history
        self.next_stable_id = 1    # Stable ID counter
        self.id_mapping = {}       # Maps tracker IDs to stable IDs
        self.max_history = max_history
        
    def update(self, tracks, frame):
        """Update with current frame's tracking results"""
        # Process all current tracks to find matches with existing objects
        current_ids = set()
        
        for track in tracks:
            tracker_id = int(track[4])
            current_ids.add(tracker_id)
            x1, y1, x2, y2 = int(track[0]), int(track[1]), int(track[2]), int(track[3])
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2
            
            # If we already know this tracker ID, update its history
            if tracker_id in self.id_mapping:
                stable_id = self.id_mapping[tracker_id]
                self.object_history[stable_id].append((cx, cy, w, h))
                # Trim history
                if len(self.object_history[stable_id]) > self.max_history:
                    self.object_history[stable_id].pop(0)
                self.tracked_objects[stable_id] = tracker_id
            else:
                # New tracker ID - try to match it with recently lost objects
                matched = False
                for stable_id, history in self.object_history.items():
                    if stable_id not in self.tracked_objects.values() and len(history) > 0:
                        # Calculate distance and size similarity to last known position
                        last_cx, last_cy, last_w, last_h = history[-1]
                        dist = ((cx - last_cx) ** 2 + (cy - last_cy) ** 2) ** 0.5
                        size_diff = abs(w * h - last_w * last_h) / (last_w * last_h + 1e-5)
                        
                        # If position and size are similar, it's likely the same object
                        if dist < 80 and size_diff < 0.3:
                            self.id_mapping[tracker_id] = stable_id
                            self.tracked_objects[stable_id] = tracker_id
                            self.object_history[stable_id].append((cx, cy, w, h))
                            matched = True
                            break
                
                # If no match found, create a new stable object
                if not matched:
                    stable_id = self.next_stable_id
                    self.next_stable_id += 1
                    self.id_mapping[tracker_id] = stable_id
                    self.tracked_objects[stable_id] = tracker_id
                    self.object_history[stable_id] = [(cx, cy, w, h)]
        
        # Remove tracker IDs that aren't present in current frame
        for tracker_id in list(self.id_mapping.keys()):
            if tracker_id not in current_ids:
                del self.id_mapping[tracker_id]
        
        # Clean up tracked_objects
        for stable_id in list(self.tracked_objects.keys()):
            if self.tracked_objects[stable_id] not in current_ids:
                self.tracked_objects.pop(stable_id)
                
    def get_stable_id(self, tracker_id):
        """Get stable ID for a given tracker ID"""
        return self.id_mapping.get(tracker_id, tracker_id)

# Pre-calculate colors for vehicle types
vehicle_colors = {
    'car': (0, 255, 0),
    'truck': (255, 0, 0),
    'bus': (0, 0, 255),
    'motorbike': (255, 255, 0)
}

# Add spatial filtering to prevent double counting
recent_crossings_up = {'car': [], 'truck': [], 'bus': [], 'motorbike': []}
recent_crossings_down = {'car': [], 'truck': [], 'bus': [], 'motorbike': []}
crossing_distance_threshold = 43  # adjusted to balance between preventing double-counts and allowing legitimate new vehicles

# FPS variables
fps_start_time = time.time()
fps_frame_count = 0
fps = 0

# Pre-calculate text positions
text_positions = {
    'up': (50, 50),
    'down': (50, 90),
    'cars_up': (250, 50),
    'trucks_up': (250, 90),
    'buses_up': (250, 130),
    'bikes_up': (250, 170),
    'cars_down': (450, 50),
    'trucks_down': (450, 90),
    'buses_down': (450, 130),
    'bikes_down': (450, 170)
}

# Set up segmentation parameters
segment_duration_sec = 120.0  # seconds per segment (5 minutes)
last_segment_start_time = 0.0
segment_number = 1
segment_events = []  # list to store crossing events for current segment

# Segment-specific counting sets to avoid duplicates within each segment
seg_carCountUp = set()
seg_truckCountUp = set()
seg_busCountUp = set()
seg_motorbikeCountUp = set()
seg_carCountDown = set()
seg_truckCountDown = set()
seg_busCountDown = set()
seg_motorbikeCountDown = set()

# Initialize enhanced tracker
enhanced_tracker = EnhancedObjectTracker()

while True:
    success, img = cap.read()
    if not success:
        break
    
    # Apply mask to frame
    if mask_3ch is not None:
        img = cv2.bitwise_and(img, mask_3ch)
    
    # Compute current video time for segmentation
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_time = current_frame / video_fps

    frame_start_time = time.time()
    
    # if imgGraphics is not None:
    #     img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    
    # Detect all classes and filter manually
    results = model(img, stream=True)
 
    # Collect detections efficiently
    detection_list = []
    detection_classes = []
    detection_sizes = []
 
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            original_cls = cls  # Save original class to use later
            if cls not in vehicle_classes:
                continue
            currentClass = classNames[cls]
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
 
            # Skip very small detections completely
            if w < MIN_DETECTION_SIZE or h < MIN_DETECTION_SIZE:
                continue
 
            # Make sure size-based classification happens first
            if currentClass == "truck" and h < MIN_TRUCK_HEIGHT:
                currentClass = "car"
            if currentClass == "bus" and h < MIN_BUS_HEIGHT:
                currentClass = "car"
                
            # Calculate confidence before using it
            conf = math.ceil((box.conf[0] * 100)) / 100
            
            # Apply class-specific confidence thresholds
            if original_cls == 3 or original_cls == 1:  # Motorbike or bicycle
                # Use lower threshold for bikes to ensure they're detected
                if conf >= MOTORBIKE_CONFIDENCE_THRESHOLD:
                    currentClass = "motorbike"
                    conf_threshold = MOTORBIKE_CONFIDENCE_THRESHOLD
                    # For bikes, we'll accept the detection even with lower confidence
                else:
                    # Lower confidence bikes default to car
                    currentClass = "car"
                    conf_threshold = CAR_CONFIDENCE_THRESHOLD
            elif currentClass == "car":
                conf_threshold = CAR_CONFIDENCE_THRESHOLD
                # For cars, require higher confidence in low-resolution/small areas
                if w < 60 or h < 60:
                    if conf < HIGH_CONFIDENCE_THRESHOLD:
                        # Skip low confidence small cars
                        continue
            elif currentClass == "truck":
                conf_threshold = TRUCK_CONFIDENCE_THRESHOLD
            elif currentClass == "bus":
                conf_threshold = BUS_CONFIDENCE_THRESHOLD
            else:
                conf_threshold = CAR_CONFIDENCE_THRESHOLD  # Default
            
            if conf > conf_threshold:
                # Let's add model's original class to the detection data
                detection_list.append([x1, y1, x2, y2, conf, original_cls])
                detection_classes.append(currentClass)
                detection_sizes.append((w, h))
    
    detections = np.array(detection_list) if detection_list else np.empty((0, 6))  # Make sure this has 6 columns for the original_cls
    
    # Update tracker
    resultsTracker = tracker.update(detections)
    
    # Update enhanced tracker
    enhanced_tracker.update(resultsTracker, current_frame)
 
    # Draw lines
    cv2.line(img, *line_coords['up'], (0, 0, 255), 5)
    cv2.line(img, *line_coords['down'], (0, 0, 255), 5)
    
    # Process tracked objects
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        # Use stable ID instead of tracker ID
        stable_id = enhanced_tracker.get_stable_id(int(id))
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2
        
        # Assign vehicle type
        if stable_id not in vehicle_first_types:
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            min_dist = float('inf')
            closest_idx = -1
            
            # Vectorized distance calculation
            if len(detections) > 0:
                det_centers = np.column_stack((
                    (detections[:, 0] + detections[:, 2]) / 2,
                    (detections[:, 1] + detections[:, 3]) / 2
                ))
                distances = np.sum((det_centers - np.array([center_x, center_y])) ** 2, axis=1)
                if len(distances) > 0:  # Make sure we have valid distances
                    closest_idx = np.argmin(distances)
                    min_dist = distances[closest_idx]
            
            # Find if this was originally detected as a motorbike by YOLO 
            was_original_bike = False
            if closest_idx >= 0 and closest_idx < len(detections):
                if detections.shape[1] >= 6:  # Check if we have the original class data
                    original_cls = int(detections[closest_idx][5])
                    if original_cls == 3 or original_cls == 1:  # 3 is motorbike, 1 is bicycle in COCO
                        was_original_bike = True
            
            if closest_idx >= 0 and closest_idx < len(detection_classes):
                current_class = detection_classes[closest_idx]
                
                # Trust YOLO for bike classification
                if was_original_bike:
                    # Make sure we have reasonable confidence in bike classification
                    # Get the original confidence from the detection
                    original_conf = 0
                    if closest_idx < len(detections):
                        original_conf = detections[closest_idx][4]
                            
                    if original_conf >= MOTORBIKE_CONFIDENCE_THRESHOLD:
                        current_class = "motorbike"
                        # Once classified as a bike, we want to maintain this classification
                        # even if the confidence drops later
                    else:
                        # Only if confidence is really low do we switch to car
                        if original_conf < (MOTORBIKE_CONFIDENCE_THRESHOLD * 0.7):  
                            current_class = "car"  # Default to car if confidence is too low
                        else:
                            # If it's close to threshold, keep as bike for consistency
                            current_class = "motorbike"

                # Apply size-based classification for consistency
                if current_class == "truck" and h < MIN_TRUCK_HEIGHT:
                    current_class = "car"
                if current_class == "bus" and h < MIN_BUS_HEIGHT:
                    current_class = "car"

                vehicle_first_types[stable_id] = current_class
                vehicleTypes[stable_id] = current_class
            else:
                # For objects with no matched detection, default to car
                vehicle_first_types[stable_id] = "car"
                vehicleTypes[stable_id] = "car"
        else:
            # IMPORTANT: Don't change vehicle type once assigned
            # This helps maintain bike classification even when objects get smaller
            vehicleTypes[stable_id] = vehicle_first_types[stable_id]
        
        vehicle_type = vehicleTypes.get(stable_id, "car")
        color = vehicle_colors.get(vehicle_type, (255, 0, 255))
        
        # Draw (lighter and faster)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)  # thinner box
        
        # Special visualization for motorbikes to aid debugging
        if vehicle_type == "motorbike":
            # Add a distinctive pattern for bikes
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Thicker yellow border
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 1)  # Diagonal line
            # Add clear identification and size information for debugging
            cv2.putText(img, f"BIKE #{stable_id} {w}x{h}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            # Draw a track line for bikes to see their movement
            if stable_id in vehicle_positions:
                prev_cx, prev_cy = vehicle_positions[stable_id]
                cv2.line(img, (prev_cx, prev_cy), (cx, cy), (255, 165, 0), 1)  # Orange trail
        # Debug small objects that might be bikes but classified as cars
        elif w < 80 and h < 130:
            cv2.putText(img, f"{w}x{h}", (cx-15, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        cv2.circle(img, (cx, cy), 3, color, cv2.FILLED)  # smaller circle
        
        vehicle_positions[stable_id] = (cx, cy)
        
        # Count using sets and record segment events (segment-specific guards)
        # For bikes, use wider boundaries to ensure they're counted
        if vehicle_type == "motorbike":
            # Use wider detection boundaries for bikes since they're smaller
            is_crossing_up = limitsUp[0] < cx < limitsUp[2] and bike_limitsUp_y_min < cy < bike_limitsUp_y_max
            is_crossing_down = limitsDown[0] < cx < limitsDown[2] and bike_limitsDown_y_min < cy < bike_limitsDown_y_max
        else:
            # Regular boundaries for other vehicles
            is_crossing_up = limitsUp[0] < cx < limitsUp[2] and limitsUp_y_min < cy < limitsUp_y_max
            is_crossing_down = limitsDown[0] < cx < limitsDown[2] and limitsDown_y_min < cy < limitsDown_y_max
            
        # Process UP direction
        if is_crossing_up:
            # For debugging bike detection at line crossing
            if vehicle_type == "motorbike":
                # Draw clear indicator when a bike is in the crossing zone
                cv2.circle(img, (cx, cy), 10, (0, 255, 255), -1)  # Yellow filled circle
                cv2.putText(img, f"CROSSING #{stable_id}", (cx-20, cy-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Check for spatial proximity to prevent double counting from ID changes
            is_new_crossing = True
            
            # Get velocity components based on previous positions if available
            vx, vy = 0, 0
            if stable_id in vehicle_positions:
                prev_cx, prev_cy = vehicle_positions[stable_id]
                vx, vy = cx - prev_cx, cy - prev_cy
            
            for prev_cx, prev_cy, prev_time in recent_crossings_up[vehicle_type]:
                dist_sq = (cx - prev_cx)**2 + (cy - prev_cy)**2
                time_diff = video_time - prev_time
                
                # Stricter check for potential double counting (same vehicle, ID changed)
                # More permissive for vehicles that appear to be following (different positions, less than 40 pixels)
                if dist_sq < (crossing_distance_threshold**2) and time_diff < 2.0:
                    # If moving in same direction as traffic flow, more likely to be a new car
                    # Check if this looks like a following vehicle versus same vehicle with ID change
                    if abs(vx) > 0 and dist_sq > 400:  # Moving and at least 20 pixels away
                        continue  # Don't mark as duplicate, likely a following vehicle
                    is_new_crossing = False
                    break
            
            if is_new_crossing:
                # Add this crossing to recent crossings list
                recent_crossings_up[vehicle_type].append((cx, cy, video_time))
                # Keep only recent crossings (last 6 seconds)
                recent_crossings_up[vehicle_type] = [(x, y, t) for x, y, t in recent_crossings_up[vehicle_type] 
                                                   if video_time - t < 6.0]
                
                # Update counters based on vehicle type
                if vehicle_type == "car" and stable_id not in seg_carCountUp:
                    seg_carCountUp.add(stable_id)
                    if stable_id not in carCountUp: 
                        carCountUp.add(stable_id)
                    cv2.line(img, *line_coords['up'], (0, 255, 0), 5)
                    segment_events.append({'Segment': segment_number, 'Vehicle_ID': stable_id, 'Type': vehicle_type, 'Direction': 'Up', 'Video_Time': round(video_time, 2)})
                elif vehicle_type == "truck" and stable_id not in seg_truckCountUp:
                    seg_truckCountUp.add(stable_id)
                    if stable_id not in truckCountUp: 
                        truckCountUp.add(stable_id)
                    cv2.line(img, *line_coords['up'], (0, 255, 0), 5)
                    segment_events.append({'Segment': segment_number, 'Vehicle_ID': stable_id, 'Type': vehicle_type, 'Direction': 'Up', 'Video_Time': round(video_time, 2)})
                elif vehicle_type == "bus" and stable_id not in seg_busCountUp:
                    seg_busCountUp.add(stable_id)
                    if stable_id not in busCountUp: 
                        busCountUp.add(stable_id)
                    cv2.line(img, *line_coords['up'], (0, 255, 0), 5)
                    segment_events.append({'Segment': segment_number, 'Vehicle_ID': stable_id, 'Type': vehicle_type, 'Direction': 'Up', 'Video_Time': round(video_time, 2)})
                elif vehicle_type == "motorbike" and stable_id not in seg_motorbikeCountUp:
                    seg_motorbikeCountUp.add(stable_id)
                    if stable_id not in motorbikeCountUp: 
                        motorbikeCountUp.add(stable_id)
                    cv2.line(img, *line_coords['up'], (0, 255, 0), 5)
                    segment_events.append({'Segment': segment_number, 'Vehicle_ID': stable_id, 'Type': vehicle_type, 'Direction': 'Up', 'Video_Time': round(video_time, 2)})
        
        if is_crossing_down:
            # Debug print statements for down line counting
            print(f"Checking DOWN: cx={cx}, cy={cy}, limits={limitsDown}, y_range=({limitsDown_y_min}, {limitsDown_y_max}), stable_id={stable_id}")
            print(f"Already in seg_carCountDown? {stable_id in seg_carCountDown}")
            print(f"recent_crossings_down: {recent_crossings_down[vehicle_type]}")
            # Check for spatial proximity to prevent double counting from ID changes
            is_new_crossing = True
            
            # Get velocity components based on previous positions if available
            vx, vy = 0, 0
            if stable_id in vehicle_positions:
                prev_cx, prev_cy = vehicle_positions[stable_id]
                vx, vy = cx - prev_cx, cy - prev_cy
            
            for prev_cx, prev_cy, prev_time in recent_crossings_down[vehicle_type]:
                dist_sq = (cx - prev_cx)**2 + (cy - prev_cy)**2
                time_diff = video_time - prev_time
                
                # Stricter check for potential double counting (same vehicle, ID changed)
                # More permissive for vehicles that appear to be following (different positions, less than 40 pixels)
                if dist_sq < (crossing_distance_threshold**2) and time_diff < 2.0:
                    # If moving in same direction as traffic flow, more likely to be a new car
                    # Check if this looks like a following vehicle versus same vehicle with ID change
                    if abs(vx) > 0 and dist_sq > 400:  # Moving and at least 20 pixels away
                        continue  # Don't mark as duplicate, likely a following vehicle
                    is_new_crossing = False
                    break
            
            if is_new_crossing:
                # Add this crossing to recent crossings list
                recent_crossings_down[vehicle_type].append((cx, cy, video_time))
                # Keep only recent crossings (last 6 seconds)
                recent_crossings_down[vehicle_type] = [(x, y, t) for x, y, t in recent_crossings_down[vehicle_type] 
                                                     if video_time - t < 6.0]
                
                # Update counters based on vehicle type
                if vehicle_type == "car" and stable_id not in seg_carCountDown:
                    seg_carCountDown.add(stable_id)
                    if stable_id not in carCountDown: 
                        carCountDown.add(stable_id)
                    cv2.line(img, *line_coords['down'], (0, 255, 0), 5)
                    segment_events.append({'Segment': segment_number, 'Vehicle_ID': stable_id, 'Type': vehicle_type, 'Direction': 'Down', 'Video_Time': round(video_time, 2)})
                elif vehicle_type == "truck" and stable_id not in seg_truckCountDown:
                    seg_truckCountDown.add(stable_id)
                    if stable_id not in truckCountDown: 
                        truckCountDown.add(stable_id)
                    cv2.line(img, *line_coords['down'], (0, 255, 0), 5)
                    segment_events.append({'Segment': segment_number, 'Vehicle_ID': stable_id, 'Type': vehicle_type, 'Direction': 'Down', 'Video_Time': round(video_time, 2)})
                elif vehicle_type == "bus" and stable_id not in seg_busCountDown:
                    seg_busCountDown.add(stable_id)
                    if stable_id not in busCountDown: 
                        busCountDown.add(stable_id)
                    cv2.line(img, *line_coords['down'], (0, 255, 0), 5)
                    segment_events.append({'Segment': segment_number, 'Vehicle_ID': stable_id, 'Type': vehicle_type, 'Direction': 'Down', 'Video_Time': round(video_time, 2)})
                elif vehicle_type == "motorbike" and stable_id not in seg_motorbikeCountDown:
                    seg_motorbikeCountDown.add(stable_id)
                    if stable_id not in motorbikeCountDown: 
                        motorbikeCountDown.add(stable_id)
                    cv2.line(img, *line_coords['down'], (0, 255, 0), 5)
                    segment_events.append({'Segment': segment_number, 'Vehicle_ID': stable_id, 'Type': vehicle_type, 'Direction': 'Down', 'Video_Time': round(video_time, 2)})
    
    # Total counts
    totalCountUp = len(carCountUp) + len(truckCountUp) + len(busCountUp) + len(motorbikeCountUp)
    totalCountDown = len(carCountDown) + len(truckCountDown) + len(busCountDown) + len(motorbikeCountDown)
    
    # Display counts using pre-calculated positions (lighter text)
    cv2.putText(img, f'Up: {totalCountUp}', text_positions['up'], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 255), 1)
    cv2.putText(img, f'Down: {totalCountDown}', text_positions['down'], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 255), 1)
    
    cv2.putText(img, f'Cars Up: {len(carCountUp)}', text_positions['cars_up'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(img, f'Trucks Up: {len(truckCountUp)}', text_positions['trucks_up'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(img, f'Buses Up: {len(busCountUp)}', text_positions['buses_up'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(img, f'Bikes Up: {len(motorbikeCountUp)}', text_positions['bikes_up'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    cv2.putText(img, f'Cars Down: {len(carCountDown)}', text_positions['cars_down'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(img, f'Trucks Down: {len(truckCountDown)}', text_positions['trucks_down'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(img, f'Buses Down: {len(busCountDown)}', text_positions['buses_down'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(img, f'Bikes Down: {len(motorbikeCountDown)}', text_positions['bikes_down'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Calculate FPS
    fps_frame_count += 1
    if fps_frame_count >= 10:
        end_time = time.time()
        fps = fps_frame_count / (end_time - fps_start_time)
        fps_start_time = time.time()
        fps_frame_count = 0
    
    cv2.putText(img, f'FPS: {fps:.1f}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
    frame_time = time.time() - frame_start_time
    cv2.putText(img, f'Frame time: {frame_time*1000:.0f}ms', (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
 
    # Check if current segment has ended (video time)
    if video_time >= last_segment_start_time + segment_duration_sec:
        if segment_events:
            df_seg = pd.DataFrame(segment_events)
            excel_file = 'traffic_segments.xlsx'
            mode = 'a' if os.path.exists(excel_file) else 'w'
            sheet_name = f"Segment_{segment_number}_{int(last_segment_start_time)}_{int(video_time)}"
            try:
                with pd.ExcelWriter(excel_file, mode=mode, engine='openpyxl') as writer:
                    df_seg.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"Segment {segment_number} saved with {len(segment_events)} events.")
            except Exception as e:
                csv_fn = f"traffic_segment_{segment_number}_{int(last_segment_start_time)}_{int(video_time)}.csv"
                df_seg.to_csv(csv_fn, index=False)
                print(f"Failed Excel write: {e}, saved to CSV: {csv_fn}")
            # Write summary CSV for this segment
            summary = {
                'Segment': segment_number,
                'Car_Up': len(seg_carCountUp),
                'Truck_Up': len(seg_truckCountUp),
                'Bus_Up': len(seg_busCountUp),
                'Bike_Up': len(seg_motorbikeCountUp),
                'Car_Down': len(seg_carCountDown),
                'Truck_Down': len(seg_truckCountDown),
                'Bus_Down': len(seg_busCountDown),
                'Bike_Down': len(seg_motorbikeCountDown)
            }
            summary_file = 'traffic_totals.csv'
            try:
                if os.path.exists(summary_file):
                    df_old = pd.read_csv(summary_file)
                    df_summary = pd.concat([df_old, pd.DataFrame([summary])], ignore_index=True)
                else:
                    df_summary = pd.DataFrame([summary])
                df_summary.to_csv(summary_file, index=False)
                print(f"Summary for segment {segment_number} saved to {summary_file}")
            except Exception as e:
                print(f"Failed to write summary CSV: {e}")
            # Write per-file totals for this segment
            per_file_totals_fn = f"traffic_segment_{segment_number}_{int(last_segment_start_time)}_{int(video_time)}_totals.csv"
            try:
                pd.DataFrame([summary]).to_csv(per_file_totals_fn, index=False)
                print(f"Per-file totals written to {per_file_totals_fn}")
            except Exception as e:
                print(f"Failed to write per-file totals CSV: {e}")
        # Reset for next segment
        segment_events = []
        seg_carCountUp.clear(); seg_truckCountUp.clear(); seg_busCountUp.clear(); seg_motorbikeCountUp.clear()
        seg_carCountDown.clear(); seg_truckCountDown.clear(); seg_busCountDown.clear(); seg_motorbikeCountDown.clear()
        last_segment_start_time += segment_duration_sec
        segment_number += 1

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After video ends, also write final segment summary
if segment_events:
    df_seg = pd.DataFrame(segment_events)
    excel_file = 'traffic_segments.xlsx'
    mode = 'a' if os.path.exists(excel_file) else 'w'
    sheet_name = f"Segment_{segment_number}_{int(last_segment_start_time)}_{int(video_time)}"
    try:
        with pd.ExcelWriter(excel_file, mode=mode, engine='openpyxl') as writer:
            df_seg.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Final segment {segment_number} saved with {len(segment_events)} events.")
    except Exception as e:
        csv_fn = f"traffic_segment_{segment_number}_{int(last_segment_start_time)}_{int(video_time)}.csv"
        df_seg.to_csv(csv_fn, index=False)
        print(f"Failed final Excel write: {e}, saved to CSV: {csv_fn}")
    # After video ends, also write final segment summary
    summary = {
        'Segment': segment_number,
        'Car_Up': len(seg_carCountUp),
        'Truck_Up': len(seg_truckCountUp),
        'Bus_Up': len(seg_busCountUp),
        'Bike_Up': len(seg_motorbikeCountUp),
        'Car_Down': len(seg_carCountDown),
        'Truck_Down': len(seg_truckCountDown),
        'Bus_Down': len(seg_busCountDown),
        'Bike_Down': len(seg_motorbikeCountDown)
    }
    summary_file = 'traffic_totals.csv'
    try:
        if os.path.exists(summary_file):
            df_old = pd.read_csv(summary_file)
            df_summary = pd.concat([df_old, pd.DataFrame([summary])], ignore_index=True)
        else:
            df_summary = pd.DataFrame([summary])
        df_summary.to_csv(summary_file, index=False)
        print(f"Final summary for segment {segment_number} saved to {summary_file}")
    except Exception as e:
        print(f"Failed to write final summary CSV: {e}")
    # Also write per-file totals for final segment
    per_file_totals_fn = f"traffic_segment_{segment_number}_{int(last_segment_start_time)}_{int(video_time)}_totals.csv"
    try:
        pd.DataFrame([summary]).to_csv(per_file_totals_fn, index=False)
        print(f"Final per-file totals written to {per_file_totals_fn}")
    except Exception as e:
        print(f"Failed to write final per-file totals CSV: {e}")

cap.release()
cv2.destroyAllWindows()
