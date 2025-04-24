from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

results = model.predict("bus.jpg", show=True)

# Keep the window open for 1 minute (60000 milliseconds)
cv2.waitKey(60000)
cv2.destroyAllWindows()


