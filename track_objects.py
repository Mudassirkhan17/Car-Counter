import cv2
from ultralytics import YOLO
import cvzone

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Open video source (0 for webcam or video file path)
cap = cv2.VideoCapture("location1.MTS")  # Change to your video file

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video source")
    exit()

# Initialize counters and tracking areas
counter = 0
tracker_history = {}

while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        print("End of video")
        break
    
    # Perform detection with tracking enabled
    results = model.track(frame, persist=True)  # Enable built-in tracking
    
    # Process detections
    if results[0].boxes is not None and hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
        for box, track_id in zip(results[0].boxes.xyxy, results[0].boxes.id):
            x1, y1, x2, y2 = box.tolist()
            track_id = int(track_id)
            
            # Get class information
            cls = int(results[0].boxes.cls[results[0].boxes.id == track_id][0])
            conf = float(results[0].boxes.conf[results[0].boxes.id == track_id][0])
            
            class_name = results[0].names[cls]
            
            # Only track vehicles (car, truck, bus, motorcycle)
            if class_name in ['car', 'truck', 'bus', 'motorcycle'] and conf > 0.3:
                # Draw bounding box and label
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f"{class_name} {track_id} {conf:.2f}", 
                                  (int(x1), int(y1)-10), scale=1, thickness=1)
                
                # Store center points for tracking
                cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                
                # Store tracking history
                if track_id not in tracker_history:
                    tracker_history[track_id] = []
                tracker_history[track_id].append((cx, cy))
                
                # Draw tracking lines
                if len(tracker_history[track_id]) > 1:
                    for i in range(1, len(tracker_history[track_id])):
                        if i % 1 == 0:  # Draw every point
                            cv2.line(frame, tracker_history[track_id][i-1], 
                                   tracker_history[track_id][i], (0, 255, 0), 2)
    
    # Display count
    cv2.putText(frame, f"Vehicle Count: {len(tracker_history)}", (20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Show the frame
    cv2.imshow("Tracking", frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows() 