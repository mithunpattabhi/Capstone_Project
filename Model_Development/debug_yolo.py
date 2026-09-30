import cv2
from ultralytics import YOLO

def debug_yolo():
    video_path = r"e:\Capstone\Videos\sample_walking.mp4"
    cap = cv2.VideoCapture(video_path)
    model = YOLO("yolov8n.pt")
    
    max_ar = 0
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        results = model(frame, verbose=False)
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:
                    x1, y1, x2, y2 = box.xyxy[0]
                    w = float(x2 - x1)
                    h = float(y2 - y1)
                    if h > 0:
                        ar = w / h
                        if ar > max_ar:
                            max_ar = ar
        frame_idx += 1
    cap.release()
    print(f"Max Aspect Ratio observed: {max_ar}")

if __name__ == "__main__":
    debug_yolo()
