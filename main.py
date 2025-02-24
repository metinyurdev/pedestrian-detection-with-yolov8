import cv2
import torch
from ultralytics import YOLO


device = "cuda" if torch.cuda.is_available() else "cpu"


model = YOLO("yolov8m.pt").to(device)  

# Define Paths
video_path = "videos/video.mp4"  
image_path = "images/image.jpg"  

# From picture
def detect_on_image(image_path):
    img = cv2.imread(image_path)
    results = model(img)

    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = f"Person {conf:.2f}"

            # Only pedestrian(Person class ID = 0)
            if int(box.cls[0]) == 0:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Pedestrian Detection - YOLOv8", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# From Video
def detect_on_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                label = f"Person {conf:.2f}"

                if int(box.cls[0]) == 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Pedestrian Detection - YOLOv8", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# Choose picture or video
#detect_on_image(image_path)  
detect_on_video(video_path)  