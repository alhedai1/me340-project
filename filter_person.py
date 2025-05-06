import torch
import cv2

# Load YOLOv5 model (you can use yolov5n.pt for smaller, faster detection)
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

# Path to your image
img_path = './classroom_students.jpg'

# Perform detection
results = model(img_path)

# The results.xywh contains the detections as [x_center, y_center, width, height, confidence, class_id]
detections = results.xywh[0].cpu().numpy()

# Filter detections for "person" (class ID 0)
person_detections = [d for d in detections if d[5] == 0]  # Class 0 is 'person'

# Print filtered detections for persons (students)
for detection in person_detections:
    x_center, y_center, w, h = detection[:4]
    confidence = detection[4]
    print(f"Detected student at (x: {x_center}, y: {y_center}), w: {w}, h: {h}, confidence: {confidence}")

# Optionally, draw bounding boxes around detected students in the image
img = cv2.imread(img_path)
for detection in person_detections:
    x_center, y_center, w, h = detection[:4]
    # Convert from YOLO's (x_center, y_center, w, h) to (x1, y1, x2, y2)
    x1, y1 = int(x_center - w / 2), int(y_center - h / 2)
    x2, y2 = int(x_center + w / 2), int(y_center + h / 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

# Display the result
cv2.imshow("Detected Students", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
