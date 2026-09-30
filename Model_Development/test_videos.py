import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch
import torch.nn.functional as F
import os
import urllib.request
from advanced_stgcn import TrueSTGCN, MP_TO_COCO_MAPPING

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ["Normal ADL", "Stimming", "Fall", "Aggression"]

#Loading the trained PyTorch TrueSTGCN model
BASE_PATH = os.path.dirname(os.path.abspath(__file__)) 
PROJECT_ROOT = os.path.dirname(BASE_PATH)
MODEL_PATH = os.path.join(PROJECT_ROOT, "Datasets_Usage", "ProcessedSkeletonDataset", "best_true_stgcn.pth")

print(f"Loading model from {MODEL_PATH}")
model = TrueSTGCN(num_classes=4).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

#Downloading the MediaPipe Tasks Model Asset
task_path = "pose_landmarker.task"
if not os.path.exists(task_path):
    print("Downloading MediaPipe tracking model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
        task_path
    )

#Modern Video Extractor using the Tasks API
def extract_skeleton_from_video(video_path, target_frames=60, target_joints=17):
    base_options = python.BaseOptions(model_asset_path=task_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0

    raw_frames_joints = []

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((frame_index / fps) * 1000)
            
            #Mediapipe requires strictly increasing timestamps
            if len(raw_frames_joints) > 0 and timestamp_ms <= int(((frame_index-1) / fps) * 1000):
                timestamp_ms = int(((frame_index-1) / fps) * 1000) + 1

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                frame_coords = np.zeros((target_joints, 3), dtype=np.float32)
                lms = result.pose_landmarks[0]

                for i, mp_idx in enumerate(MP_TO_COCO_MAPPING):
                    frame_coords[i] = [lms[mp_idx].x, lms[mp_idx].y, lms[mp_idx].z]

                raw_frames_joints.append(frame_coords)

            frame_index += 1

    cap.release()

    if len(raw_frames_joints) == 0:
        raise ValueError(f"No human skeleton detected in {video_path}")

    raw_array = np.array(raw_frames_joints, dtype=np.float32)
    T_raw = raw_array.shape[0]

    #THE TEMPORAL FIX: Find the most active 2-second (60 frame) window
    if T_raw <= target_frames:
        start_idx = 0
        end_idx = T_raw
    else:
        max_var = -1
        start_idx = 0
        y_coords = raw_array[:, :, 1] # Track vertical motion (Y-axis)
        for i in range(T_raw - target_frames + 1):
            window_var = np.var(y_coords[i:i+target_frames])
            if window_var > max_var:
                max_var = window_var
                start_idx = i
        end_idx = start_idx + target_frames

    data_window = raw_array[start_idx:end_idx]

    #Transpose to (Channels, Frames, Joints, Persons) -> (3, T, 17, 1)
    data = np.transpose(data_window, (2, 0, 1))
    data = np.expand_dims(data, axis=-1)
    
    #Center and Scale Normalization
    hip_center = (data[:, :, 11, 0] + data[:, :, 12, 0]) / 2.0  # Shape: (3, T)
    data[:, :, :, 0] = data[:, :, :, 0] - hip_center[:, :, np.newaxis]
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val

    #Temporal Interpolation (Match Training)
    T_curr = data.shape[1]
    if T_curr != target_frames:
        new_data = np.zeros((3, target_frames, target_joints, 1), dtype=np.float32)
        orig_indices = np.linspace(0, T_curr - 1, num=target_frames)
        for c in range(3):
            for v in range(target_joints):
                new_data[c, :, v, 0] = np.interp(orig_indices, np.arange(T_curr), data[c, :, v, 0])
        data = new_data

    return data

#End-to-End Prediction wrapper
def predict_video(video_path):
    print(f"\nProcessing Video: {video_path}")
    try:
        skeleton = extract_skeleton_from_video(video_path)
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return
        
    tensor_in = torch.from_numpy(skeleton).float().unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor_in)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))

    print("=" * 45)
    print(f"PREDICTED: {class_names[pred_idx].upper()} ({probs[pred_idx]*100:.1f}%)")
    print("=" * 45)
    for i, name in enumerate(class_names):
        bar = "#" * int(probs[i] * 20)
        print(f"{name:<12}: {probs[i]*100:5.1f}% | {bar}")


#Downloading and Testing the Sample Videos
print("\nDownloading sample videos...")
walk_path = "sample_walking.mp4"
if not os.path.exists(walk_path):
    urllib.request.urlretrieve("https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/face-demographics-walking.mp4", walk_path)

fall_path = "sample_falling.mp4"
if not os.path.exists(fall_path):
    urllib.request.urlretrieve("http://fenix.ur.edu.pl/~mkepski/ds/data/fall-01-cam0.mp4", fall_path)

fall2_path = "sample_falling_2.mp4"
if not os.path.exists(fall2_path):
    urllib.request.urlretrieve("http://fenix.ur.edu.pl/~mkepski/ds/data/fall-02-cam0.mp4", fall2_path)

adl_path = "sample_sitting.mp4"
if not os.path.exists(adl_path):
    urllib.request.urlretrieve("http://fenix.ur.edu.pl/~mkepski/ds/data/adl-01-cam0.mp4", adl_path)

predict_video(walk_path)
predict_video(fall_path)
predict_video(fall2_path)
predict_video(adl_path)
predict_video("sample-test.mp4")
predict_video("sample-test1.mp4")
