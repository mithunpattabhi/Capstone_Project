import librosa
import numpy as np
from moviepy import VideoFileClip
import os

def check_volumes():
    vids = [
        r"e:\Capstone\Videos\Man_Fall_With_Audio.mp4",
        r"e:\Capstone\Videos\sample_walking.mp4",
        r"e:\Capstone\Videos\sample_sitting.mp4",
        r"e:\Capstone\Videos\sample_test.mp4", 
        r"e:\Capstone\Videos\sample_test1.mp4"
    ]
    for v in vids:
        if not os.path.exists(v): continue
        try:
            clip = VideoFileClip(v)
            if clip.audio is None:
                print(f"{os.path.basename(v)}: NO AUDIO")
                continue
            clip.audio.write_audiofile("temp.wav", logger=None)
            y, sr = librosa.load("temp.wav", sr=22050)
            rms = librosa.feature.rms(y=y)[0]
            print(f"{os.path.basename(v)}: Max RMS = {np.max(rms):.4f}")
        except Exception as e:
            pass

if __name__ == "__main__":
    check_volumes()
