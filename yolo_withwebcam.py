import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")


cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 640)

while True:
    ret, frame = cap.read()
    results = model(frame,stream=True)
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


