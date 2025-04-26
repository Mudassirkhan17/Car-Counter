import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import time
import os

# Set fixed output frame size for faster processing
TARGET_WIDTH = 640
TARGET_HEIGHT = 480
 
# Load video
cap = cv2.VideoCapture("00008.MTS")  # For Video

# Use a smaller model for better performance
model = YOLO("yolov8s.pt")  # Using smaller model (s instead of l)

# Configure YOLO for faster inference
model.conf = 0.25  # Higher confidence threshold = fewer detections = faster
model.iou = 0.45   # Intersection over union threshold
 
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

# Set target classes for filtering after detection (not during)
target_classes = ["car", "truck", "bus", "motorbike"]
 
# Load and resize mask once at the beginning - with error handling
try:
    mask_original = cv2.imread("mask.png")
    if mask_original is None:
        print("Warning: mask.png not found. Using full frame.")
        mask = np.ones((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8) * 255
    else:
        mask = cv2.resize(mask_original, (TARGET_WIDTH, TARGET_HEIGHT))
        if len(mask.shape) == 2:  # If mask is grayscale
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
except Exception as e:
    print(f"Error loading mask: {e}")
    mask = np.ones((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8) * 255

# Load graphics once - with error handling
try:
    imgGraphics_original = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    if imgGraphics_original is None:
        print("Warning: graphics.png not found. No overlay will be applied.")
        imgGraphics = None
    else:
        imgGraphics = cv2.resize(imgGraphics_original, (TARGET_WIDTH, TARGET_HEIGHT))
except Exception as e:
    print(f"Error loading graphics: {e}")
    imgGraphics = None
 
# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Default counting line (can be modified by user)
limits = [int(400 * TARGET_WIDTH / 1280), int(297 * TARGET_HEIGHT / 720), 
          int(673 * TARGET_WIDTH / 1280), int(297 * TARGET_HEIGHT / 720)]

# Variables for line setting mode
is_setting_line = False
temp_limits = limits.copy()
line_points = []

# Create separate counts for different vehicle types
carCount = []
truckCount = []
busCount = []
motorbikeCount = []

# Create a dictionary to keep track of vehicle types by ID
vehicleTypes = {}

# Performance tracking
frame_count = 0
start_time = time.time()
fps = 0

# Define allowed classes for faster filtering after prediction
target_class_indices = [classNames.index(cls) for cls in target_classes]

# Mouse callback function for setting the counting line
def set_line(event, x, y, flags, param):
    global line_points, temp_limits, is_setting_line, limits
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(line_points) < 2:
            line_points.append((x, y))
            print(f"Point {len(line_points)} set at ({x}, {y})")
            
            if len(line_points) == 2:
                temp_limits = [line_points[0][0], line_points[0][1], 
                               line_points[1][0], line_points[1][1]]
                print(f"New line coordinates: {temp_limits}")
                limits = temp_limits
                line_points = []
                is_setting_line = False
                print("Line setting completed. Press 'l' again if you want to set a new line.")

# Create a window and set mouse callback
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", set_line)

while True:
    try:
        success, img_original = cap.read()
        if not success:
            print("End of video or error reading frame")
            break
        
        # Resize frame for faster processing
        img = cv2.resize(img_original, (TARGET_WIDTH, TARGET_HEIGHT))
        
        # Only process frames if not in line setting mode
        if not is_setting_line:
            # Apply the mask
            imgRegion = cv2.bitwise_and(img, mask)
            
            # Overlay graphics (if available)
            if imgGraphics is not None:
                try:
                    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
                except Exception as e:
                    print(f"Error overlaying graphics: {e}")
            
            # Run YOLO detection - WITHOUT filtering by class
            # Let YOLO detect all objects and we'll filter afterwards
            results = model(imgRegion, stream=True)
        
            detections = np.empty((0, 5))
            detection_classes = {}
        
    for r in results:
        boxes = r.boxes
        for box in boxes:
                    # Get class and filter after detection
                    cls = int(box.cls[0])
                    if cls not in target_class_indices:
                        continue  # Skip non-vehicle classes
                        
                    # Bounding Box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
                    # Confidence
                    conf = float(box.conf[0])
                    
                    # Class Name
                    currentClass = classNames[cls]
        
                    if conf > 0.25:  # Already filtered by model.conf but keeping as a check
                        currentArray = np.array([x1, y1, x2, y2, conf])
                        
                        # Store this detection's class
                        idx = len(detections)
                        detection_classes[idx] = currentClass
                        
                        detections = np.vstack((detections, currentArray))
        
            # Update tracker with new detections
            resultsTracker = tracker.update(detections)
        
            # Process tracked objects
            for result in resultsTracker:
                x1, y1, x2, y2, id = result
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                
                # Try to associate this tracking ID with a vehicle type only if new
                if int(id) not in vehicleTypes:
                    # Find the closest detection using vectorized operations for speed
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                    
                    # More efficient - only process a detection if there's at least one
                    if len(detections) > 0:
                        det_centers_x = (detections[:, 0] + detections[:, 2]) / 2
                        det_centers_y = (detections[:, 1] + detections[:, 3]) / 2
                        
                        # Vectorized distance calculation
                        distances = np.sqrt((det_centers_x - center_x)**2 + (det_centers_y - center_y)**2)
                        closest_idx = np.argmin(distances)
                        
                        if closest_idx in detection_classes:
                            vehicleTypes[int(id)] = detection_classes[closest_idx]
                        else:
                            vehicleTypes[int(id)] = "unknown"
                    else:
                        vehicleTypes[int(id)] = "unknown"
                
                # Get the vehicle type for this tracked object
                vehicle_type = vehicleTypes.get(int(id), "unknown")
                
                # Color coding based on vehicle type - create color lookup table for speed
                color_map = {
                    "car": (0, 255, 0),      # Green
                    "truck": (255, 0, 0),    # Blue
                    "bus": (0, 0, 255),      # Red
                    "motorbike": (255, 255, 0), # Cyan
                    "unknown": (255, 0, 255)  # Magenta
                }
                color = color_map.get(vehicle_type, (255, 0, 255))
                
                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Simpler text display - faster than cvzone.putTextRect
                cv2.putText(img, f'{vehicle_type} {int(id)}', (max(0, x1), max(35, y1)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
                cx, cy = x1 + w // 2, y1 + h // 2
                cv2.circle(img, (cx, cy), 5, color, cv2.FILLED)
        
                # Check for counting line crossing
                if limits[0] < cx < limits[2] and (limits[1] - 15) < cy < (limits[1] + 15):
                    # Check if this ID has already been counted
                    count_arrays = {
                        "car": carCount,
                        "truck": truckCount,
                        "bus": busCount,
                        "motorbike": motorbikeCount
                    }
                    
                    # Only update if this vehicle type is in our tracking and not already counted
                    if vehicle_type in count_arrays and id not in count_arrays[vehicle_type]:
                        count_arrays[vehicle_type].append(id)
                        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 2)
            
            # Calculate total count
            totalCount = len(carCount) + len(truckCount) + len(busCount) + len(motorbikeCount)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count >= 10:  # Update FPS every 10 frames
                end_time = time.time()
                fps = frame_count / (end_time - start_time)
                frame_count = 0
                start_time = time.time()
            
            # Display counts and FPS
            cv2.putText(img, f'FPS: {fps:.1f}', (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(img, f'Total: {totalCount}', (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 255), 2)
            cv2.putText(img, f'Cars: {len(carCount)}', (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f'Trucks: {len(truckCount)}', (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(img, f'Buses: {len(busCount)}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(img, f'Motorbikes: {len(motorbikeCount)}', (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Draw counting line
        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 2)
        
        # Show instructions when in line setting mode
        if is_setting_line:
            cv2.putText(img, "Click to set the first point", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if len(line_points) == 1:
                cv2.putText(img, "Click to set the second point", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Draw the first point
                cv2.circle(img, line_points[0], 5, (0, 0, 255), -1)
        
        # Show current line coordinates
        cv2.putText(img, f"Line: {limits}", (50, TARGET_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Image", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
        break
        elif key == ord('l'):  # Press 'l' to set a new line
            print("Line setting mode activated. Click to set the counting line.")
            is_setting_line = True
            line_points = []
    
    except Exception as e:
        print(f"Error processing frame: {e}")
        continue

cap.release()
cv2.destroyAllWindows()