from flask import Flask, request, jsonify, render_template
import os
import sys
import numpy as np
import torch
import cv2
import traceback

# Add Model dir to path so we can import advanced_stgcn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Model')))

from advanced_stgcn import TrueSTGCN, AudioFeatureExtractor, MultimodalFusionSystem, MP_TO_COCO_MAPPING
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
try:
    from moviepy import VideoFileClip
except ImportError:
    print("Warning: moviepy not installed properly")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ["Normal ADL", "Stimming", "Fall", "Aggression"]

print("Pre-loading models (this takes a moment)...")
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Model', 'best_true_stgcn.pth'))
stgcn_model = TrueSTGCN(num_classes=4).to(device)
if os.path.exists(MODEL_PATH):
    stgcn_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
stgcn_model.eval()

fusion_system = MultimodalFusionSystem(stgcn_model=stgcn_model, volume_threshold=0.14, enable_yolo=False)
audio_extractor = AudioFeatureExtractor()

task_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Model', 'pose_landmarker.task'))
mp_base_options = python.BaseOptions(model_asset_path=task_path)
mp_options = vision.PoseLandmarkerOptions(
    base_options=mp_base_options,
    running_mode=vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
        
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_video.mp4')
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_audio.wav')
    file.save(video_path)

    try:
        # Extract Audio
        max_energy = 0
        mfccs_mean = np.zeros((13,))
        try:
            clip = VideoFileClip(video_path)
            clip.audio.write_audiofile(audio_path, logger=None)
            max_energy, mfccs_mean = audio_extractor.extract_features(audio_path)
            clip.close()
        except Exception as e:
            print(f"Audio extraction failed: {e}")

        # Extract frames and skeleton
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps): fps = 30.0

        raw_frames_joints = []
        video_frames = []

        with vision.PoseLandmarker.create_from_options(mp_options) as landmarker:
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

        # Build skeleton tensor
        if len(raw_frames_joints) == 0:
            data = np.zeros((3, 60, 17, 1), dtype=np.float32)
        else:
            raw_array = np.array(raw_frames_joints, dtype=np.float32)
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
        audio_features = (max_energy, mfccs_mean)

        # Predict
        brightness = fusion_system.calculate_brightness(video_frames)
        final_pred, final_probs, yolo_triggered = fusion_system.infer(
            skeleton_data=skeleton_tensor,
            video_frames=video_frames,
            audio_features=audio_features,
            device=device
        )

        final_class = class_names[final_pred]
        mode = "Day Mode"
        if brightness < 30: mode = "Night Mode"
        elif brightness < 80: mode = "Dusk Mode"

        audio_override = max_energy > (fusion_system.volume_threshold * 2.0)

        response = {
            'success': True,
            'prediction': final_class,
            'confidence': f"{final_probs[final_pred]*100:.1f}%",
            'brightness': f"{brightness:.1f}",
            'mode': mode,
            'audio_detected': bool(max_energy > 0),
            'audio_volume': f"{max_energy:.2f}",
            'audio_override_triggered': bool(audio_override),
            'yolo_override_triggered': bool(yolo_triggered),
        }
        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
