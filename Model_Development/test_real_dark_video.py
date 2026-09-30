import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch
import torch.nn.functional as F
import os
import urllib.request
import scipy.io.wavfile as wav

from advanced_stgcn import TrueSTGCN, AudioFeatureExtractor, MultimodalFusionSystem, MP_TO_COCO_MAPPING

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ["Normal ADL", "Stimming", "Fall", "Aggression"]

def test_real_dark_video():
    # 1. Get the video
    video_path = "sample_falling.mp4"
    if not os.path.exists(video_path):
        print("Downloading sample falling video...")
        urllib.request.urlretrieve("http://fenix.ur.edu.pl/~mkepski/ds/data/fall-01-cam0.mp4", video_path)

    # 2. Download MediaPipe tracking model if needed
    task_path = "pose_landmarker.task"
    if not os.path.exists(task_path):
        urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task", task_path)

    print("\n--- 1. Processing Video (Simulating Dark Environment) ---")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 30.0

    raw_frames_joints = []
    darkened_video_frames = []

    base_options = python.BaseOptions(model_asset_path=task_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # Simulating pitch-black darkness by dropping brightness 
            dark_frame = (frame * 0.05).astype(np.uint8)
            darkened_video_frames.append(dark_frame)

            # NOTE: We run Mediapipe on the ORIGINAL frame just to get *some* skeleton data 
            # to feed the STGCN (if we ran it on the dark frame, it would completely fail to find a person). 
            # In real life, an IR camera might provide the skeleton, but the RGB feed would be dark.
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

    # Process skeleton into expected tensor shape
    raw_array = np.array(raw_frames_joints, dtype=np.float32)
    start_idx = 0
    end_idx = min(len(raw_array), 60)
    data_window = raw_array[start_idx:end_idx]
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

    print("\n--- 2. Simulating Extracted Audio (Loud Thud/Fall) ---")
    # Because UR Fall videos don't have audio tracks, we will generate a loud sound 
    # that simulates the mic picking up a heavy fall in the dark ward.
    sample_rate = 22050
    t = np.linspace(0, 1, sample_rate)
    fall_sound = np.sin(2 * np.pi * 50 * t) * np.exp(-5 * t) # A low-frequency thud
    wav.write("simulated_fall_thud.wav", sample_rate, fall_sound.astype(np.float32))
    
    print("\n--- 3. Running Multimodal Fusion ---")
    BASE_PATH = os.path.dirname(os.path.abspath(__file__)) 
    PROJECT_ROOT = os.path.dirname(BASE_PATH)
    MODEL_PATH = os.path.join(PROJECT_ROOT, "Datasets_Usage", "ProcessedSkeletonDataset", "best_true_stgcn.pth")
    
    # Load trained model
    stgcn_model = TrueSTGCN(num_classes=4).to(device)
    if os.path.exists(MODEL_PATH):
        stgcn_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("Warning: Trained STGCN model not found. Using untrained weights for demonstration.")
        
    # Setup Fusion
    extractor = AudioFeatureExtractor()
    audio_features = extractor.extract_features("simulated_fall_thud.wav")
    fusion_system = MultimodalFusionSystem(stgcn_model=stgcn_model, volume_threshold=0.2)

    # 1. See what the raw STGCN predicts (it might be confused because the skeleton is weird or untrained)
    stgcn_model.eval()
    with torch.no_grad():
        raw_logits = stgcn_model(skeleton_tensor)
        raw_probs = torch.softmax(raw_logits, dim=1).cpu().numpy()[0]
    
    print("\n[Visual Only Prediction (STGCN)]")
    for i, name in enumerate(class_names):
        print(f"   {name:<12}: {raw_probs[i]*100:5.1f}%")

    # 2. See what the Late Fusion system decides
    brightness = fusion_system.calculate_brightness(darkened_video_frames)
    final_pred, final_probs = fusion_system.infer(
        skeleton_data=skeleton_tensor,
        video_frames=darkened_video_frames,
        audio_features=audio_features,
        device=device
    )

    print("\n[Multimodal Fusion Prediction]")
    print(f"   -> Average Room Brightness: {brightness:.2f}/255")
    if brightness < 30:
        print("   -> [NIGHT MODE] Pitch-Black conditions detected! Heavily weighting Audio Sensors (80%).")
        
    print("\nFinal Decision:")
    for i, name in enumerate(class_names):
        indicator = " <--- WINNER" if i == final_pred else ""
        print(f"   {name:<12}: {final_probs[i]*100:5.1f}%{indicator}")

if __name__ == "__main__":
    test_real_dark_video()
