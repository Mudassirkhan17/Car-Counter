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
vehicle_classes = set([2, 3, 5, 7])  # car, motorcycle, bus, truck

# Define minimum size thresholds for vehicle types
MIN_TRUCK_HEIGHT = 140
MIN_BUS_HEIGHT = 160

# Confidence thresholds
CONFIDENCE_THRESHOLD = 0.3
MOTORBIKE_CONFIDENCE_THRESHOLD = 0.25

# Load graphics image once
imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
if imgGraphics is None:
    print("Warning: Could not load graphics.png, continuing without overlay")
    imgGraphics = None

# Initialize tracker with optimized parameters
tracker = Sort(max_age=35, min_hits=3, iou_threshold=0.3)

# Define counting lines
# limitsUp = [614, 396, 1038, 387]
limitsUp = [260-15, 382-35, 610-15, 389-35]
# limitsDown = [514+100, 396+40, 1038+100, 387+40]
limitsDown = [153-15, 389+20, 610-15, 399+20]

# Pre-calculate boundary ranges
limitsUp_y_min = limitsUp[1] - 22
limitsUp_y_max = limitsUp[1] + 22
limitsDown_y_min = limitsDown[1] - 22 
limitsDown_y_max = limitsDown[1] + 22

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

# Pre-calculate colors for vehicle types
vehicle_colors = {
    'car': (0, 255, 0),
    'truck': (255, 0, 0),
    'bus': (0, 0, 255),
    'motorbike': (255, 255, 0)
}

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
segment_duration_sec = 60.0  # seconds per segment
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

while True:
    success, img = cap.read()
    if not success:
        break
    
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
            if cls not in vehicle_classes:
                continue
            currentClass = classNames[cls]
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
 
            # Size-based classification
            if currentClass == "truck" and h < MIN_TRUCK_HEIGHT:
                currentClass = "car"
            if currentClass == "bus" and h < MIN_BUS_HEIGHT:
                currentClass = "car"
                
            conf = math.ceil((box.conf[0] * 100)) / 100
            conf_threshold = MOTORBIKE_CONFIDENCE_THRESHOLD if currentClass == "motorbike" else CONFIDENCE_THRESHOLD
            
            if conf > conf_threshold:
                detection_list.append([x1, y1, x2, y2, conf])
                detection_classes.append(currentClass)
                detection_sizes.append((w, h))
    
    detections = np.array(detection_list) if detection_list else np.empty((0, 5))
    
    # Update tracker
    resultsTracker = tracker.update(detections)
 
    # Draw lines
    cv2.line(img, *line_coords['up'], (0, 0, 255), 5)
    cv2.line(img, *line_coords['down'], (0, 0, 255), 5)
    
    # Process tracked objects
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2
        
        # Assign vehicle type
        if int(id) not in vehicle_first_types:
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
                closest_idx = np.argmin(distances)
                min_dist = distances[closest_idx]
            
            if closest_idx >= 0:
                current_class = detection_classes[closest_idx]
                if current_class == "truck" and h < MIN_TRUCK_HEIGHT:
                    current_class = "car"
                if current_class == "bus" and h < MIN_BUS_HEIGHT:
                    current_class = "car"
                vehicle_first_types[int(id)] = current_class
                vehicleTypes[int(id)] = current_class
            else:
                vehicle_first_types[int(id)] = "car"
                vehicleTypes[int(id)] = "car"
        else:
            vehicleTypes[int(id)] = vehicle_first_types[int(id)]
        
        vehicle_type = vehicleTypes.get(int(id), "car")
        color = vehicle_colors.get(vehicle_type, (255, 0, 255))
        
        # Draw (lighter and faster)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)  # thinner box
        label = f'{vehicle_type} {int(id)}'
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]  # smaller font, thinner
        cv2.rectangle(img, (x1, y1 - t_size[1] - 6), (x1 + t_size[0], y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.circle(img, (cx, cy), 3, color, cv2.FILLED)  # smaller circle
        
        vehicle_positions[int(id)] = (cx, cy)
        
        # Count using sets and record segment events (segment-specific guards)
        if limitsUp[0] < cx < limitsUp[2] and limitsUp_y_min < cy < limitsUp_y_max:
            if vehicle_type == "car" and id not in seg_carCountUp:
                seg_carCountUp.add(id)
                if id not in carCountUp: carCountUp.add(id)
                cv2.line(img, *line_coords['up'], (0, 255, 0), 5)
                segment_events.append({'Segment': segment_number, 'Vehicle_ID': int(id), 'Type': vehicle_type, 'Direction': 'Up', 'Video_Time': round(video_time, 2)})
            elif vehicle_type == "truck" and id not in seg_truckCountUp:
                seg_truckCountUp.add(id)
                if id not in truckCountUp: truckCountUp.add(id)
                cv2.line(img, *line_coords['up'], (0, 255, 0), 5)
                segment_events.append({'Segment': segment_number, 'Vehicle_ID': int(id), 'Type': vehicle_type, 'Direction': 'Up', 'Video_Time': round(video_time, 2)})
            elif vehicle_type == "bus" and id not in seg_busCountUp:
                seg_busCountUp.add(id)
                if id not in busCountUp: busCountUp.add(id)
                cv2.line(img, *line_coords['up'], (0, 255, 0), 5)
                segment_events.append({'Segment': segment_number, 'Vehicle_ID': int(id), 'Type': vehicle_type, 'Direction': 'Up', 'Video_Time': round(video_time, 2)})
            elif vehicle_type == "motorbike" and id not in seg_motorbikeCountUp:
                seg_motorbikeCountUp.add(id)
                if id not in motorbikeCountUp: motorbikeCountUp.add(id)
                cv2.line(img, *line_coords['up'], (0, 255, 0), 5)
                segment_events.append({'Segment': segment_number, 'Vehicle_ID': int(id), 'Type': vehicle_type, 'Direction': 'Up', 'Video_Time': round(video_time, 2)})
        if limitsDown[0] < cx < limitsDown[2] and limitsDown_y_min < cy < limitsDown_y_max:
            if vehicle_type == "car" and id not in seg_carCountDown:
                seg_carCountDown.add(id)
                if id not in carCountDown: carCountDown.add(id)
                cv2.line(img, *line_coords['down'], (0, 255, 0), 5)
                segment_events.append({'Segment': segment_number, 'Vehicle_ID': int(id), 'Type': vehicle_type, 'Direction': 'Down', 'Video_Time': round(video_time, 2)})
            elif vehicle_type == "truck" and id not in seg_truckCountDown:
                seg_truckCountDown.add(id)
                if id not in truckCountDown: truckCountDown.add(id)
                cv2.line(img, *line_coords['down'], (0, 255, 0), 5)
                segment_events.append({'Segment': segment_number, 'Vehicle_ID': int(id), 'Type': vehicle_type, 'Direction': 'Down', 'Video_Time': round(video_time, 2)})
            elif vehicle_type == "bus" and id not in seg_busCountDown:
                seg_busCountDown.add(id)
                if id not in busCountDown: busCountDown.add(id)
                cv2.line(img, *line_coords['down'], (0, 255, 0), 5)
                segment_events.append({'Segment': segment_number, 'Vehicle_ID': int(id), 'Type': vehicle_type, 'Direction': 'Down', 'Video_Time': round(video_time, 2)})
            elif vehicle_type == "motorbike" and id not in seg_motorbikeCountDown:
                seg_motorbikeCountDown.add(id)
                if id not in motorbikeCountDown: motorbikeCountDown.add(id)
                cv2.line(img, *line_coords['down'], (0, 255, 0), 5)
                segment_events.append({'Segment': segment_number, 'Vehicle_ID': int(id), 'Type': vehicle_type, 'Direction': 'Down', 'Video_Time': round(video_time, 2)})
    
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