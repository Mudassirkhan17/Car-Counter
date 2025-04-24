import cv2
from ultralytics import YOLO
import math
import cvzone


model = YOLO("yolov8n.pt")

# Load the mask
mask = cv2.imread("mask.png")


classname = ["Person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("cars.mp4")
cap.set(3, 640)
cap.set(4, 640)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Resize the mask to match the frame dimensions
    resized_mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    
    # Ensure both images have the same number of channels
    if len(resized_mask.shape) == 2:  # If mask is grayscale
        resized_mask = cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2BGR)
    
    # Now perform the bitwise_and operation
    imageRegion = cv2.bitwise_and(frame, resized_mask)
    
    results = model(frame, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            w,h = x2-x1,y2-y1

            conf = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])

            currentclass = classname[cls]
            if currentclass == "car" or currentclass == "truck" or currentclass == "bus" or currentclass == "motorbike" and conf > 0.3:
                cvzone.putTextRect(frame,f"{currentclass} {conf}",(max(0,x1),max(35,y1)),scale=1,thickness=1,offset=3)
                cvzone.cornerRect(frame,(x1,y1,w,h),l=4,rt=2)

                # current_count += 1

            # cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),5)
            
    cv2.imshow("YOLO", frame)
    cv2.imshow("Image Region", imageRegion)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


