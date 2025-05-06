import cv2
import torch

# constants
# mock classroom dimensions (cm)
width = 110
length = 90
height = 40
fov = 70 
dist_to_front_row = 30
left_camera_pos = 25
center_camera_pos = 55
right_camera_pos = 85

camera_indices = [1, 2, 3] # 0 is laptop's camera
image_files = ["left.jpg", "center.jpg", "right.jpg"]
# lists of detections, each detection = [x_center, y_center, width, height, confidence, class]
left_detections = []
center_detections = []
right_detections = []
person_detections = [[], [], []]


# get list of indices of available cameras (laptop is 0)
def list_available_cameras(max=5):
    available = []
    for index in range(max):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available.append(index)
            cap.release()
    return available

# capture & save 3 images
def capture_images():
    for idx, cam_idx in enumerate(camera_indices):
        cap = cv2.VideoCapture(cam_idx)

        if not cap.isOpened():
            print(f"Camera {cam_idx} not accessible.")
            continue

        ret, frame = cap.read()
        if ret:
            cv2.imwrite(image_files[idx], frame)
            print(f"Saved image from camera {cam_idx} as {image_files[idx]}")
        else:
            print(f"Failed to capture image from camera {cam_idx}")

        cap.release()


def detect_people():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
    img_paths = image_files
    for idx, img_path in enumerate(img_paths):
        results = model(img_path)
        detections = results.xywh[0].cpu().numpy()
        person_detections[idx] = [d for d in detections if d[5] == 0]  # Class 0 is 'person'

        # for detection in person_detections[idx]:
        #     x_center, y_center, w, h = detection[:4]
        #     confidence = detection[4]
        #     print(f"Detected student at (x: {x_center}, y: {y_center}), w: {w}, h: {h}, confidence: {confidence}")

    #     img = cv2.imread(img_path)
    #     for detection in person_detections:
    #         x_center, y_center, w, h = detection[:4]
    #         x1, y1 = int(x_center - w / 2), int(y_center - h / 2)
    #         x2, y2 = int(x_center + w / 2), int(y_center + h / 2)
    #         cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    # cv2.imshow("Detected Students", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return person_detections

def draw_bboxes(img_path, detections):
    img = cv2.imread(img_path)
    for detection in detections:
        x_c, y_c, w, h = detection[:4]
        x1, y1 = int(x_c - w/2), int(y_c - h/2)
        x2, y2 = int(x_c + w/2), int(y_c + h/2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imshow(img_path, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # cameras = list_available_cameras(max=5)
    # print(f"Detected cameras: {cameras}")
    # capture_images()
    person_detections = detect_people()
    left_detections = person_detections[0]
    center_detections = person_detections[1]
    right_detections = person_detections[2]

    print(left_detections)
    draw_bboxes(image_files[0], left_detections)
    draw_bboxes(image_files[1], center_detections)
    draw_bboxes(image_files[2], right_detections)
