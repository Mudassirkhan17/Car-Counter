import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import time
 
cap = cv2.VideoCapture("location1.MTS")  # For Video
 
model = YOLO("yolov8n.pt")
# Set model to CPU mode explicitly
model.to('cpu')

# classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            
#               ]
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
# mask = cv2.imread("mask.png")

# Define minimum size thresholds for vehicle types
MIN_TRUCK_HEIGHT = 140  # Minimum height for an object to be classified as a truck
MIN_BUS_HEIGHT = 160    # Minimum height for an object to be classified as a bus

# Confidence thresholds - lower for motorbikes to improve detection
CONFIDENCE_THRESHOLD = 0.3  # General confidence threshold
MOTORBIKE_CONFIDENCE_THRESHOLD = 0.25  # Lower threshold for motorbikes
 
# Cache the graphics image to avoid loading it every frame
imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
if imgGraphics is None:
    print("Warning: Could not load graphics.png, continuing without overlay")
    imgGraphics = None

# Tracking
tracker = Sort(max_age=35, min_hits=3, iou_threshold=0.3)
 
# Define three counting lines
# To increase length: decrease first x-coordinate and increase second x-coordinate
# Original: limitsUp = [614-560, 396, 1038-560, 387]
# Find center: (614-560 + 1038-560)/2 = 266
# Add 100px to each side
limitsUp = [614, 396, 1038, 387]    # Upper boundary - extended by 100px on each side
limitsDown = [153-15, 389+15, 599-15,399+9]   # Lower boundary - parallel to Up line, 100px lower
limitsLeft = [0, 0, 0, 0]   # Left boundary - fixed to be a vertical line

# Pre-calculate boundary ranges for faster detection
limitsUp_y_min = limitsUp[1] - 22
limitsUp_y_max = limitsUp[1] + 22
limitsDown_y_min = limitsDown[1] - 22 
limitsDown_y_max = limitsDown[1] + 22
limitsLeft_x_min = limitsLeft[0] - 22
limitsLeft_x_max = limitsLeft[0] + 14
 
# Create separate counts for different vehicle types
carCountUp = []
truckCountUp = []
busCountUp = []
motorbikeCountUp = []

carCountDown = []
truckCountDown = []
busCountDown = []
motorbikeCountDown = []

# Add counts for left direction
carCountLeft = []
truckCountLeft = []
busCountLeft = []
motorbikeCountLeft = []

# Create a dictionary to keep track of vehicle types by ID
vehicleTypes = {}
# Track position history to prevent multiple counts
vehicle_positions = {}
# Store the first detected vehicle type for each ID
vehicle_first_types = {}

# FPS calculation variables
fps_start_time = time.time()
fps_frame_count = 0
fps = 0
 
while True:
    success, img = cap.read()
    if not success:
        break
    
    # Start timing this frame for FPS calculation
    frame_start_time = time.time()
    
    # Add the graphics overlay only if loaded successfully
    if imgGraphics is not None:
        img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    
    results = model(img, stream=True)
 
    detections = np.empty((0, 5))
    
    # Dictionary to map detection boxes to their class
    detection_classes = {}
    # Store the dimensions of each detection 
    detection_sizes = {}
 
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get class index
            cls = int(box.cls[0])
            
            # Safety check for class index
            if cls >= len(classNames):
                print(f"Warning: Invalid class index {cls}")
                continue
                
            # Class Name
            currentClass = classNames[cls]
            
            # Only process vehicle classes we're interested in
            if currentClass not in ["car", "truck", "bus", "motorbike"]:
                continue
                
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            
            # Size-based classification correction - before adding to detection list
            # Apply strict size rules to prevent misclassification
            if currentClass == "truck" and h < MIN_TRUCK_HEIGHT:
                currentClass = "car"  # Reclassify as car if too small for a truck
            if currentClass == "bus" and h < MIN_BUS_HEIGHT:
                currentClass = "car"  # Reclassify as car if too small for a bus
 
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            
            # Only add high confidence detections - use lower threshold for motorbikes
            conf_threshold = MOTORBIKE_CONFIDENCE_THRESHOLD if currentClass == "motorbike" else CONFIDENCE_THRESHOLD
            if conf > conf_threshold:
                currentArray = np.array([x1, y1, x2, y2, conf])
                
                # Store this detection's class before adding to the detections array
                idx = len(detections)
                detection_classes[idx] = currentClass
                detection_sizes[idx] = (w, h)
                
                detections = np.vstack((detections, currentArray))
 
    # Update tracker with new detections
    resultsTracker = tracker.update(detections)
 
    # Draw counting lines
    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)
    cv2.line(img, (limitsLeft[0], limitsLeft[1]), (limitsLeft[2], limitsLeft[3]), (0, 0, 255), 5)
    
    # Process tracked objects
    for i, result in enumerate(resultsTracker):
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        
        # Calculate center of the object
        cx, cy = x1 + w // 2, y1 + h // 2
        
        # Try to associate this tracking ID with a vehicle type
        # If the ID is new, we'll use the closest detection to determine its class
        if int(id) not in vehicle_first_types:
            # Find the closest detection to this tracked object's position
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            min_dist = float('inf')
            closest_idx = -1
            
            # Check all detections to find the closest one
            for idx, det_class in detection_classes.items():
                if idx < len(detections):
                    det_x1, det_y1, det_x2, det_y2 = detections[idx, :4]
                    det_center_x = (det_x1 + det_x2) / 2
                    det_center_y = (det_y1 + det_y2) / 2
                    
                    # Calculate distance between centers
                    dist = ((det_center_x - center_x) ** 2 + (det_center_y - center_y) ** 2) ** 0.5
                    
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = idx
            
            # If we found a close match, use its class
            if closest_idx >= 0 and closest_idx in detection_classes:
                # Get the current detected class
                current_class = detection_classes[closest_idx]
                
                # Apply additional size check with current object dimensions
                if current_class == "truck" and h < MIN_TRUCK_HEIGHT:
                    current_class = "car"  # Force it to be a car
                if current_class == "bus" and h < MIN_BUS_HEIGHT:
                    current_class = "car"  # Force it to be a car
                
                # Store the first type permanently
                vehicle_first_types[int(id)] = current_class
                vehicleTypes[int(id)] = current_class
            else:
                # Default to "car" if no match found - safer than "unknown"
                vehicle_first_types[int(id)] = "car"
                vehicleTypes[int(id)] = "car"
        else:
            # Always use the first assigned type
            vehicleTypes[int(id)] = vehicle_first_types[int(id)]
        
        # Get the vehicle type for this tracked object
        vehicle_type = vehicleTypes.get(int(id), "car")  # Default to car instead of unknown
        
        # Color coding based on vehicle type
        color = (255, 0, 255)  # Default color (magenta)
        if vehicle_type == "car":
            color = (0, 255, 0)  # Green
        elif vehicle_type == "truck":
            color = (255, 0, 0)  # Blue
        elif vehicle_type == "bus":
            color = (0, 0, 255)  # Red
        elif vehicle_type == "motorbike":
            color = (255, 255, 0)  # Cyan
        
        # Simplified drawing (faster than cvzone functions)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Simplified text rendering (faster than cvzone.putTextRect)
        label = f'{vehicle_type} {int(id)}'
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(img, (x1, y1-t_size[1]-10), (x1+t_size[0], y1), color, -1)
        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
 
        cv2.circle(img, (cx, cy), 5, color, cv2.FILLED)
        
        # Store last position of this vehicle
        vehicle_positions[int(id)] = (cx, cy)
 
        # Check for upper counting line crossing
        if limitsUp[0] < cx < limitsUp[2] and limitsUp_y_min < cy < limitsUp_y_max:
            # Check if this ID has already been counted for the upper line
            if vehicle_type == "car" and id not in carCountUp:
                # Spatial filter - check if there's already a similar vehicle in a nearby position
                too_close = False
                for count_id in carCountUp[-5:]:  # Check last 5 added cars
                    if count_id in vehicle_positions:
                        # Get the position of the already counted vehicle
                        count_x, count_y = vehicle_positions[count_id]
                        # Calculate distance between this vehicle and the counted one
                        dist = ((cx - count_x) ** 2 + (cy - count_y) ** 2) ** 0.5
                        if dist < 70:  # If closer than 70 pixels, likely the same vehicle
                            too_close = True
                            break
                
                if not too_close:
                    carCountUp.append(id)
                    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)
            elif vehicle_type == "truck" and id not in truckCountUp:
                truckCountUp.append(id)
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)
            elif vehicle_type == "bus" and id not in busCountUp:
                busCountUp.append(id)
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)
            elif vehicle_type == "motorbike" and id not in motorbikeCountUp:
                motorbikeCountUp.append(id)
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)
        
        # Check for lower counting line crossing
        if limitsDown[0] < cx < limitsDown[2] and limitsDown_y_min < cy < limitsDown_y_max:
            # Check if this ID has already been counted for the lower line
            if vehicle_type == "car" and id not in carCountDown:
                # Spatial filter - check if there's already a similar vehicle in a nearby position
                too_close = False
                for count_id in carCountDown[-5:]:  # Check last 5 added cars
                    if count_id in vehicle_positions:
                        # Get the position of the already counted vehicle
                        count_x, count_y = vehicle_positions[count_id]
                        # Calculate distance between this vehicle and the counted one
                        dist = ((cx - count_x) ** 2 + (cy - count_y) ** 2) ** 0.5
                        if dist < 70:  # If closer than 70 pixels, likely the same vehicle
                            too_close = True
                            break
                
                if not too_close:
                    carCountDown.append(id)
                    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)
            elif vehicle_type == "truck" and id not in truckCountDown:
                truckCountDown.append(id)
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)
            elif vehicle_type == "bus" and id not in busCountDown:
                busCountDown.append(id)
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)
            elif vehicle_type == "motorbike" and id not in motorbikeCountDown:
                motorbikeCountDown.append(id)
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)
        
        # Check for left boundary crossing
        if limitsLeft_x_min < cx < limitsLeft_x_max and limitsLeft[1] < cy < limitsLeft[3]:
            # Check if this ID has already been counted for the left line
            if vehicle_type == "car" and id not in carCountLeft:
                # Spatial filter - check if there's already a similar vehicle in a nearby position
                too_close = False
                for count_id in carCountLeft[-5:]:  # Check last 5 added cars
                    if count_id in vehicle_positions:
                        # Get the position of the already counted vehicle
                        count_x, count_y = vehicle_positions[count_id]
                        # Calculate distance between this vehicle and the counted one
                        dist = ((cx - count_x) ** 2 + (cy - count_y) ** 2) ** 0.5
                        if dist < 70:  # If closer than 70 pixels, likely the same vehicle
                            too_close = True
                            break
                
                if not too_close:
                    carCountLeft.append(id)
                    cv2.line(img, (limitsLeft[0], limitsLeft[1]), (limitsLeft[2], limitsLeft[3]), (0, 255, 0), 5)
            elif vehicle_type == "truck" and id not in truckCountLeft:
                truckCountLeft.append(id)
                cv2.line(img, (limitsLeft[0], limitsLeft[1]), (limitsLeft[2], limitsLeft[3]), (0, 255, 0), 5)
            elif vehicle_type == "bus" and id not in busCountLeft:
                busCountLeft.append(id)
                cv2.line(img, (limitsLeft[0], limitsLeft[1]), (limitsLeft[2], limitsLeft[3]), (0, 255, 0), 5)
            elif vehicle_type == "motorbike" and id not in motorbikeCountLeft:
                motorbikeCountLeft.append(id)
                cv2.line(img, (limitsLeft[0], limitsLeft[1]), (limitsLeft[2], limitsLeft[3]), (0, 255, 0), 5)
 
    # Calculate total counts
    totalCountUp = len(carCountUp) + len(truckCountUp) + len(busCountUp) + len(motorbikeCountUp)
    totalCountDown = len(carCountDown) + len(truckCountDown) + len(busCountDown) + len(motorbikeCountDown)
    totalCountLeft = len(carCountLeft) + len(truckCountLeft) + len(busCountLeft) + len(motorbikeCountLeft)
    
    # Display counts with bolder text and shorter labels
    # Direction totals
    cv2.putText(img, f'Up: {totalCountUp}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 255), 2)
    cv2.putText(img, f'Down: {totalCountDown}', (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 255), 2)
    cv2.putText(img, f'Left: {totalCountLeft}', (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 255), 2)
    
    # Up counts - use shorter text and bolder font
    cv2.putText(img, f'Cars Up: {len(carCountUp)}', (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img, f'Trucks Up: {len(truckCountUp)}', (250, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(img, f'Buses Up: {len(busCountUp)}', (250, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, f'Bikes Up: {len(motorbikeCountUp)}', (250, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Down counts
    cv2.putText(img, f'Cars Down: {len(carCountDown)}', (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img, f'Trucks Down: {len(truckCountDown)}', (450, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(img, f'Buses Down: {len(busCountDown)}', (450, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, f'Bikes Down: {len(motorbikeCountDown)}', (450, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Left counts
    cv2.putText(img, f'Cars Left: {len(carCountLeft)}', (650, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img, f'Trucks Left: {len(truckCountLeft)}', (650, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(img, f'Buses Left: {len(busCountLeft)}', (650, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, f'Bikes Left: {len(motorbikeCountLeft)}', (650, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Calculate FPS
    fps_frame_count += 1
    if fps_frame_count >= 10:  # Update FPS every 10 frames
        end_time = time.time()
        fps = fps_frame_count / (end_time - fps_start_time)
        fps_start_time = time.time()
        fps_frame_count = 0
    
    # Display FPS
    cv2.putText(img, f'FPS: {fps:.1f}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Calculate frame processing time
    frame_time = time.time() - frame_start_time
    cv2.putText(img, f'Frame time: {frame_time*1000:.0f}ms', (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()