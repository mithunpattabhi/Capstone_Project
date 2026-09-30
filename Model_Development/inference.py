import os
import cv2
import torch
import argparse
import numpy as np
import mediapipe as mp
from collections import deque
from advanced_stgcn import TrueSTGCN, MP_TO_COCO_MAPPING

# CONFIGURATION
CLASSES = {
    0: ("Normal", (0, 255, 0)),        # Green
    1: ("Stimming", (0, 255, 255)),    # Yellow
    2: ("Fall", (0, 0, 255)),          # Red
    3: ("Aggression", (255, 165, 0))   # Orange
}

TARGET_FRAMES = 60
TARGET_JOINTS = 17
SMOOTHING_WINDOW = 15 # Wait for 15 frames of consistent prediction before changing alert state

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='Video file path or camera index (default: 0)')
    return parser.parse_args()

#INFERENCE PIPELINE
def main():
    args = get_args()
    
    #Setup Device & Loading Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading True ST-GCN on {device}...")
    
    model = TrueSTGCN(num_classes=4, in_channels=3).to(device)
    
    #Resolving Model Path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "Datasets_Usage", "ProcessedSkeletonDataset", "best_true_stgcn.pth")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Could not find trained weights at {model_path}!")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully!")
    
    #Setup MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    #Setup Video Source
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
        
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Could not open video source {source}")
        return
        
    #Buffers
    frame_buffer = deque(maxlen=TARGET_FRAMES)
    prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)
    current_alert = 0 # Normal
    
    print("\nStarting Real-Time Inference... Press 'q' to quit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        #Flip frame horizontally for a selfie-view display (if webcam)
        if isinstance(source, int):
            frame = cv2.flip(frame, 1)
            
        h, w, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        #Process MediaPipe Pose
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            #Extract 33 joints (X, Y, Z)
            joints = np.zeros((33, 3), dtype=np.float32)
            for i, lm in enumerate(results.pose_landmarks.landmark):
                joints[i] = [lm.x, lm.y, lm.z]
                
            #Map 33 -> 17 (COCO Format used in our ST-GCN)
            joints_17 = joints[MP_TO_COCO_MAPPING]
            frame_buffer.append(joints_17)
            
            #Predict if we have enough frames
            if len(frame_buffer) == TARGET_FRAMES:
                # Build Tensor: shape (C, T, V, M) -> (3, 60, 17, 1)
                seq_data = np.array(frame_buffer) # (60, 17, 3)
                seq_data = np.transpose(seq_data, (2, 0, 1)) # (3, 60, 17)
                
                #Center and Scale
                hip_center = (seq_data[:, :, 11] + seq_data[:, :, 12]) / 2.0
                seq_data = seq_data - hip_center[:, :, np.newaxis]
                max_val = np.max(np.abs(seq_data))
                if max_val > 0:
                    seq_data = seq_data / max_val
                    
                seq_data = np.expand_dims(seq_data, axis=-1) # (3, 60, 17, 1)
                
                #Create batch dimension
                input_tensor = torch.from_numpy(seq_data).float().unsqueeze(0).to(device)
                
                #Inference
                with torch.no_grad():
                    output = model(input_tensor)
                    _, predicted = torch.max(output, 1)
                    raw_pred = predicted.item()
                    
                #Temporal Smoothing (Must be consistent for SMOOTHING_WINDOW frames)
                prediction_buffer.append(raw_pred)
                
                #Count occurrences in the buffer
                counts = np.bincount(list(prediction_buffer))
                smoothed_pred = np.argmax(counts)
                
                #Only change alert if the smoothed prediction differs and is dominant enough
                if counts[smoothed_pred] >= (SMOOTHING_WINDOW // 2):
                    current_alert = smoothed_pred
                    
        #Render Alert Text
        label, color = CLASSES.get(current_alert, ("Unknown", (255, 255, 255)))
        
        #Drawing a nice background box for the text
        cv2.rectangle(frame, (10, 10), (350, 70), (0, 0, 0), -1)
        cv2.putText(frame, f"STATUS: {label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)
        
        #If it's an emergency class, draw a warning border around the screen
        if current_alert in [2, 3]: # Fall or Aggression
            cv2.rectangle(frame, (0, 0), (w, h), color, 10)
            
        cv2.imshow("ST-GCN Autism Caregiver Alert System", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
