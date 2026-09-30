import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch
import os
import sys

# Import moviepy for safe audio extraction on Windows without ffmpeg globally
try:
    from moviepy import VideoFileClip
except ImportError as e:
    print(f"Warning: moviepy not installed properly: {e}")

from advanced_stgcn import TrueSTGCN, AudioFeatureExtractor, MultimodalFusionSystem, MP_TO_COCO_MAPPING

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ["Normal ADL", "Stimming", "Fall", "Aggression"]

def process_user_video():
    video_path = r"e:\Capstone_Project\Man_Fall_With_Audio.mp4"
    audio_temp_path = r"e:\Capstone_Project\temp_audio.wav"

    print("\n--- 1. Extracting Audio Track from Video ---")
    try:
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_temp_path, logger=None)
        print("   -> Audio track successfully extracted to WAV.")
    except Exception as e:
        print(f"Failed to extract audio: {e}")
        return

    print("\n--- 2. Processing Video Frames & Skeleton ---")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 30.0

    task_path = "pose_landmarker.task"
    base_options = python.BaseOptions(model_asset_path=task_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    raw_frames_joints = []
    video_frames = []

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            video_frames.append(frame)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            timestamp_ms = int((frame_index / fps) * 1000)
            if len(raw_frames_joints) > 0 and timestamp_ms <= int(((frame_index-1) / fps) * 1000):
                timestamp_ms = int(((frame_index-1) / fps) * 1000) + 1

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                frame_coords = np.zeros((17, 3), dtype=np.float32)
                lms = result.pose_landmarks[0]
                for i, mp_idx in enumerate(MP_TO_COCO_MAPPING):
                    frame_coords[i] = [lms[mp_idx].x, lms[mp_idx].y, lms[mp_idx].z]
                raw_frames_joints.append(frame_coords)
            frame_index += 1
    cap.release()

    if len(raw_frames_joints) == 0:
        print("   -> WARNING: No skeleton could be detected in this video at all! (Is it too dark?)")
        # We will feed zeros to the visual model
        data = np.zeros((3, 60, 17, 1), dtype=np.float32)
    else:
        # Process skeleton into expected tensor shape
        raw_array = np.array(raw_frames_joints, dtype=np.float32)
        
        # Simple windowing - grab most active part
        T_raw = raw_array.shape[0]
        if T_raw <= 60:
            start_idx = 0
        else:
            max_var = -1
            start_idx = 0
            y_coords = raw_array[:, :, 1]
            for i in range(T_raw - 60 + 1):
                window_var = np.var(y_coords[i:i+60])
                if window_var > max_var:
                    max_var = window_var
                    start_idx = i

        data_window = raw_array[start_idx:start_idx+60]
        data = np.transpose(data_window, (2, 0, 1))
        data = np.expand_dims(data, axis=-1)

        hip_center = (data[:, :, 11, 0] + data[:, :, 12, 0]) / 2.0
        data[:, :, :, 0] = data[:, :, :, 0] - hip_center[:, :, np.newaxis]
        max_val = np.max(np.abs(data))
        if max_val > 0: data = data / max_val
        
        T_curr = data.shape[1]
        if T_curr != 60:
            new_data = np.zeros((3, 60, 17, 1), dtype=np.float32)
            orig_indices = np.linspace(0, T_curr - 1, num=60)
            for c in range(3):
                for v in range(17):
                    new_data[c, :, v, 0] = np.interp(orig_indices, np.arange(T_curr), data[c, :, v, 0])
            data = new_data
            
    skeleton_tensor = torch.from_numpy(data).float().unsqueeze(0).to(device)

    print("\n--- 3. Running Multimodal Fusion ---")
    BASE_PATH = os.path.dirname(os.path.abspath(__file__)) 
    PROJECT_ROOT = os.path.dirname(BASE_PATH)
    MODEL_PATH = os.path.join(PROJECT_ROOT, "Datasets_Usage", "ProcessedSkeletonDataset", "best_true_stgcn.pth")
    
    stgcn_model = TrueSTGCN(num_classes=4).to(device)
    if os.path.exists(MODEL_PATH):
        stgcn_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        
    # Setup Fusion
    extractor = AudioFeatureExtractor()
    audio_features = extractor.extract_features(audio_temp_path)
    # Using volume threshold of 0.1 for real-world audio, as it might not be a pure sine wave loud noise
    fusion_system = MultimodalFusionSystem(stgcn_model=stgcn_model, volume_threshold=0.1)

    stgcn_model.eval()
    with torch.no_grad():
        raw_logits = stgcn_model(skeleton_tensor)
        raw_probs = torch.softmax(raw_logits, dim=1).cpu().numpy()[0]
    
    print("\n[Visual Only Prediction (STGCN)]")
    for i, name in enumerate(class_names):
        print(f"   {name:<12}: {raw_probs[i]*100:5.1f}%")

    brightness = fusion_system.calculate_brightness(video_frames)
    final_pred, final_probs, yolo_triggered = fusion_system.infer(
        skeleton_data=skeleton_tensor,
        video_frames=video_frames,
        audio_features=audio_features,
        device=device
    )

    print("\n[Multimodal Fusion Prediction]")
    if yolo_triggered:
        print("   -> [YOLO ENSEMBLE OVERRIDE] Person detected lying horizontal! Visual score forced to FALL.")
    print(f"   -> Average Room Brightness: {brightness:.2f}/255")
    
    max_energy = audio_features[0] if audio_features is not None else 0
    if max_energy > (fusion_system.volume_threshold * 2.0):
        print("   -> [CRITICAL AUDIO OVERRIDE] Exceptionally loud sound detected! Overriding daylight visual sensors.")
    elif brightness < 30:
        print("   -> [NIGHT MODE] Low-light conditions detected! Weighting Audio Sensors heavily.")
    elif brightness < 80:
        print("   -> [DUSK MODE] Dim lighting detected. Splitting weight 50/50.")
    else:
        print("   -> [DAY MODE] Good lighting detected. Trusting Visual Sensors primarily.")
        
    print("\nFinal Decision:")
    for i, name in enumerate(class_names):
        indicator = " <--- WINNER" if i == final_pred else ""
        print(f"   {name:<12}: {final_probs[i]*100:5.1f}%{indicator}")

if __name__ == "__main__":
    process_user_video()
