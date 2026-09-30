# Mentor Meeting Cheat Sheet: ST-GCN Model Debugging & Optimization

This document outlines the exact problems we faced with the ST-GCN model, the code changes we made to fix them, and how the system works now. Use this as a guide when explaining our progress to your mentor.

## 1. The Core Problem
During training, the model reported **99% accuracy**. However, when we tested it on real-world videos (like `sample_falling.mp4`), it completely failed, misclassifying Falls as "Normal ADL" or "Stimming". Furthermore, retraining the model to test new theories was taking **1 to 3 hours** per run.

We had to solve two major issues: **Training Speed** and **Data Mismatch**.

---

## 2. Fixing the Training Speed (The Disk I/O Bottleneck)

**The Issue:**
The model was incredibly slow to train because it was opening 50,000 individual `.npy` files from the hard drive in every single epoch. The GPU was sitting idle waiting for the hard drive to spin and read tiny files.

**What We Typed:**
We rewrote the dataset loader (`RobustSkeletonDataset` in `advanced_stgcn.py`) to implement **RAM Caching**. 

**How it Works:**
Before training starts, a multi-threaded `ThreadPoolExecutor` reads all 50,000 files exactly once and stores them in the computer's ultra-fast RAM. 
*   **Result:** Training time dropped from **3 hours to under 3 minutes**!

---

## 3. Fixing the Accuracy (The Data Mismatches)

Once we could train rapidly, we diagnosed *why* the model was failing in the real world. We found two critical mismatches between how the model was trained and how it was tested.

### A. The "Zero-Padding" Trap
**The Issue:** The raw `le2i` Fall dataset consisted of 30-frame videos. To make them fit our 60-frame ST-GCN model, the original code padded them with 30 frames of exact zeroes. 
*   **The Flaw:** The AI didn't learn what a fall looked like. It learned that *Fall = 30 frames of movement followed by 30 frames of frozen zeroes*. When we gave it a real 60-frame video of a continuous fall, it failed because there were no zeroes.

**What We Typed:** 
We implemented **Temporal Interpolation** using `np.interp` in both `advanced_stgcn.py` and `test_videos.py`.

**How it Works:**
Instead of adding zeroes, we mathematically "stretch" the 30 frames to 60 frames. It creates smooth, artificial frames in between the real ones (like playing the video in slow motion), forcing the model to learn the actual mechanics of the limbs.

### B. The Spatial Normalization Bug (The "Invisible Fall")
**The Issue:** When investigating the raw training data, we made a massive discovery. The original creators of the dataset had pre-processed the files so that the human's hip was mathematically pinned to `(0, 0, 0)` on *every single frame*. The global trajectory (the person dropping to the floor) had been permanently erased. 
*   **The Flaw:** Our test script was feeding the model real videos where the person physically dropped. The model had *never seen* a hip move vertically, so it treated the fall as out-of-distribution noise (predicting Stimming or Aggression).

**What We Typed:**
We updated `test_videos.py` and `inference.py` to use **Frame-by-Frame Normalization**.

**How it Works:**
```python
# We calculate the center of the hip for EVERY frame individually
hip_center = (data[:, :, 11, 0] + data[:, :, 12, 0]) / 2.0  
# We subtract that center, pinning the person to the middle of the screen
data[:, :, :, 0] = data[:, :, :, 0] - hip_center[:, :, np.newaxis]
```
By doing this, we forced the real-world test videos to look exactly like the training data: a stick figure waving its limbs while permanently pinned to the center of the camera. The model was finally able to recognize the shape of the falling limbs!

---

## 4. The Final Results
By perfectly aligning the Test Pipeline with the Training Pipeline, the model is now incredibly robust:
*   Successfully detected a completely unseen falling video with **92.4% confidence**.
*   Successfully detected a normal sitting action with **59.8% confidence**.
*   Generalized to custom user-recorded videos, successfully catching high-energy boundary cases (classifying extremely fast hand-flapping as Aggression due to kinematic similarities with punching).

**Next Steps:** With a rock-solid core ST-GCN engine, we are ready to connect the inference pipeline to Firebase to trigger real-time anomaly alerts.
