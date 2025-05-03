import cv2
import numpy as np

# ... existing code up through mask thresholding ...
_, mask_bin = cv2.threshold(mask_img, 128, 255, cv2.THRESH_BINARY)

# --- compute tight ROI once from mask ---
ys, xs = np.where(mask_bin > 0)
if xs.size and ys.size:
    roi_x1, roi_x2 = xs.min(), xs.max()
    roi_y1, roi_y2 = ys.min(), ys.max()
else:
    roi_x1, roi_y1, roi_x2, roi_y2 = 0, 0, mask_bin.shape[1], mask_bin.shape[0]
print(f"Computed ROI from mask: x[{roi_x1}:{roi_x2}], y[{roi_y1}:{roi_y2}]")
# ... rest of initialization ...

while True:
    success, img = cap.read()
    if not success:
        break

    # Apply mask to full frame
    masked = cv2.bitwise_and(img, img, mask=mask_bin)

    # --- crop to ROI before detection ---
    masked_roi = masked[roi_y1:roi_y2, roi_x1:roi_x2]

    # detect on the smaller ROI
    results = model(masked_roi, stream=True)

    # Collect detections efficiently
    detection_list = []
    detection_classes = []
    detection_sizes = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls not in vehicle_classes:
                continue
            # get ROI-relative coords, then shift back
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1) + roi_x1, int(y1) + roi_y1, int(x2) + roi_x1, int(y2) + roi_y1
            w, h = x2 - x1, y2 - y1

            # Size-based classification, confidence checks, etc.
            # ... unchanged logic ...
            if conf > conf_threshold:
                detection_list.append([x1, y1, x2, y2, conf])
                detection_classes.append(currentClass)
                detection_sizes.append((w, h))

    # ... rest of loop stays the same, using full-frame coords for drawing and tracking ... 