import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
 
cap = cv2.VideoCapture("00008.MTS")  # For Video
 
model = YOLO("yolov8s.pt")

# Define which classes we want to detect
target_classes = ["car", "truck", "bus", "motorbike"]
 
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

# Pre-calculate indices of target classes for faster filtering
target_class_indices = []
for cls in target_classes:
    if cls in classNames:
        target_class_indices.append(classNames.index(cls))
 
mask = cv2.imread("mask.png")
 
# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
 
limits = [400, 297, 673, 297]

# Create separate counts for different vehicle types
carCount = []
truckCount = []
busCount = []
motorbikeCount = []

# Create a dictionary to keep track of vehicle types by ID
vehicleTypes = {}
 
while True:
    success, img = cap.read()
    if not success:
        break
    
    # Resize mask to match frame dimensions
    resized_mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    
    # Ensure both have same number of channels
    if len(resized_mask.shape) == 2:  # If mask is grayscale
        resized_mask = cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2BGR)
    
    # Now perform the bitwise_and operation with correctly sized mask
    imgRegion = cv2.bitwise_and(img, resized_mask)
 
    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    results = model(imgRegion, stream=True)
 
    detections = np.empty((0, 5))
    
    # Dictionary to map detection boxes to their class
    detection_classes = {}
 
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get class index
            cls = int(box.cls[0])
            
            # Skip if not a target vehicle class
            if cls not in target_class_indices:
                continue
                
            # Safely check class name (prevent IndexError)
            if 0 <= cls < len(classNames):
                currentClass = classNames[cls]
            else:
                print(f"Warning: Invalid class index {cls}")
                continue
                
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
 
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            
            # Only process if confidence is good
            if conf > 0.1:
                currentArray = np.array([x1, y1, x2, y2, conf])
                
                # Store this detection's class before adding to the detections array
                idx = len(detections)
                detection_classes[idx] = currentClass
                
                detections = np.vstack((detections, currentArray))
 
    # Update tracker with new detections
    resultsTracker = tracker.update(detections)
 
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    
    # Process tracked objects
    for i, result in enumerate(resultsTracker):
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        
        # Try to associate this tracking ID with a vehicle type
        # If the ID is new, we'll use the closest detection to determine its class
        if int(id) not in vehicleTypes:
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
                vehicleTypes[int(id)] = detection_classes[closest_idx]
            else:
                # Default to "unknown" if no match found
                vehicleTypes[int(id)] = "unknown"
        
        # Get the vehicle type for this tracked object
        vehicle_type = vehicleTypes.get(int(id), "unknown")
        
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
        
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=color)
        cvzone.putTextRect(img, f'{vehicle_type} {int(id)}', (max(0, x1), max(35, y1)),
                           scale=1, thickness=2, offset=5, colorR=color)
 
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, color, cv2.FILLED)
 
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            # Check if this ID has already been counted
            if vehicle_type == "car" and id not in carCount:
                carCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
            elif vehicle_type == "truck" and id not in truckCount:
                truckCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
            elif vehicle_type == "bus" and id not in busCount:
                busCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
            elif vehicle_type == "motorbike" and id not in motorbikeCount:
                motorbikeCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
 
    # Calculate total count
    totalCount = len(carCount) + len(truckCount) + len(busCount) + len(motorbikeCount)
    
    # Display counts for each vehicle type
    cv2.putText(img, f'Total: {totalCount}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)
    cv2.putText(img, f'Cars: {len(carCount)}', (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f'Trucks: {len(truckCount)}', (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(img, f'Buses: {len(busCount)}', (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, f'Motorbikes: {len(motorbikeCount)}', (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
 
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 