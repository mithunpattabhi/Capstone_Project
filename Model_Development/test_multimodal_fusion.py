import os
import torch
import numpy as np
import scipy.io.wavfile as wav
import sys

# Import the classes we just added
from advanced_stgcn import TrueSTGCN, AudioFeatureExtractor, MultimodalFusionSystem

def generate_dummy_data():
    print("1. Generating dummy data for testing...")
    
    # 1. Dummy Audio (Loud noise / "Anomaly" sound)
    # 2 seconds of audio at 22050 Hz
    sample_rate = 22050
    t = np.linspace(0, 2, sample_rate * 2)
    # A loud 440Hz sine wave (will trigger high volume / energy)
    audio_data = np.sin(2 * np.pi * 440 * t) 
    audio_path = "test_loud_noise.wav"
    wav.write(audio_path, sample_rate, audio_data.astype(np.float32))
    print(f"   -> Created {audio_path}")

    # 2. Dummy Skeleton Data (BatchSize=1, Channels=3, Frames=60, Joints=17, Persons=1)
    # This matches the TrueSTGCN input shape (N, C, T, V, M) expected in infer()
    skeleton_data = torch.rand(1, 3, 60, 17, 1)
    print("   -> Created dummy skeleton tensor: shape", skeleton_data.shape)

    # 3. Dummy Video Frames (Low Light condition)
    # Simulate 3 frames of 224x224 RGB video that are extremely dark
    # Pixel values around 15 (out of 255)
    video_frames = [np.full((224, 224, 3), 15, dtype=np.uint8) for _ in range(3)]
    print("   -> Created dummy low-light video frames.")
    
    return audio_path, skeleton_data, video_frames

def run_test():
    audio_path, skeleton_data, video_frames = generate_dummy_data()
    
    print("\n2. Initializing Models...")
    # Initialize our visual model (untrained, just testing the pipeline)
    stgcn_model = TrueSTGCN(num_classes=4)
    stgcn_model.eval()
    
    # Initialize the audio extractor
    extractor = AudioFeatureExtractor()
    
    # Initialize our Fusion system
    fusion_system = MultimodalFusionSystem(stgcn_model=stgcn_model, volume_threshold=0.3)
    
    print("\n3. Extracting Audio Features...")
    try:
        max_energy, mfccs_mean = extractor.extract_features(audio_path)
        print(f"   -> Max Energy Detected: {max_energy:.4f}")
        print(f"   -> MFCCs Shape: {mfccs_mean.shape}")
    except Exception as e:
        print("   -> Audio extraction failed:", e)
        return

    print("\n4. Running Multimodal Late Fusion...")
    try:
        brightness = fusion_system.calculate_brightness(video_frames)
        print(f"   -> Detected Room Brightness: {brightness:.2f}/255 (Dark room detected? {brightness < 30})")
        
        audio_features = (max_energy, mfccs_mean)
        final_pred, final_probs = fusion_system.infer(
            skeleton_data=skeleton_data, 
            video_frames=video_frames, 
            audio_features=audio_features, 
            device='cpu'
        )
        
        print(f"\n[SUCCESS] Pipeline completed without errors.")
        print("-" * 40)
        print(f"Final Class Prediction: {final_pred}")
        print(f"Final Blended Probabilities (Normal, Stimming, Fall, Aggression):")
        print(np.round(final_probs, 4))
        print("-" * 40)
        
    except Exception as e:
        print("\n[ERROR] Pipeline failed during inference:")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
