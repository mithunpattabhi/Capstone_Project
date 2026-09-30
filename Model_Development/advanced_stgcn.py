import os
import glob
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import zipfile

# --- Audio Integration Imports ---
try:
    import librosa
    import cv2
except ImportError:
    pass # Handled at runtime if used

# CONFIGURATION

RUNNING_ON_COLAB = False

if RUNNING_ON_COLAB:
    ZIP_DIR = "/content/drive/MyDrive/Capstone_Project/capstone_datasets/skeletons"
    WORKING_DIR = "/content/drive/MyDrive/Capstone_Project/Datasets_Usage/skeletons_unzipped"
    OUTPUT_DIR = "/content/drive/MyDrive/Capstone_Project/Datasets_Usage/ProcessedSkeletonDataset"
else:
    try:
        BASE_PATH = os.path.dirname(os.path.abspath(__file__)) 
    except NameError:
        BASE_PATH = os.getcwd()
        
    PROJECT_ROOT = os.path.dirname(BASE_PATH)
    ZIP_DIR = os.path.join(PROJECT_ROOT, "Dataset", "skeletons")
    WORKING_DIR = os.path.join(PROJECT_ROOT, "Datasets_Usage", "skeletons_unzipped")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Datasets_Usage", "ProcessedSkeletonDataset")

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

#GRAPH DEFINITION (COCO 17 Joints)
#Since the largest dataset (ASDPose) uses 17 joints (COCO), 
#we map all 33-joint MediaPipe data down to 17 joints to preserve topological integrity!
class Graph:
    def __init__(self):
        self.num_node = 17
        self.edges = [
            (0,1), (1,3), (0,2), (2,4), # Face
            (5,7), (7,9), (6,8), (8,10), # Arms
            (5,6), (5,11), (6,12), (11,12), # Torso
            (11,13), (13,15), (12,14), (14,16) # Legs
        ]
        self.A = self.get_adjacency_matrix()
    
    def get_adjacency_matrix(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edges:
            A[i, j] = 1
            A[j, i] = 1
        for i in range(self.num_node):
            A[i, i] = 1
        
        #Normalize Adjacency Matrix
        D = np.sum(A, axis=1)
        D = np.diag(D**-0.5)
        A = D @ A @ D
        return np.expand_dims(A, axis=0) # Shape: (1, 17, 17)

#ST-GCN ARCHITECTURE
class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, s_kernel_size=1):
        super().__init__()
        self.s_kernel_size = s_kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels * s_kernel_size, kernel_size=1)
        
    def forward(self, x, A):
        x = self.conv(x)
        N, KC, T, V = x.size()
        x = x.view(N, self.s_kernel_size, KC // self.s_kernel_size, T, V)
        x = torch.einsum('nkctv,kvw->nctw', x, A)
        return x.contiguous()

class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1))
        
    def forward(self, x):
        return self.conv(x)

class STGCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, residual=True):
        super().__init__()
        self.gcn = SpatialGraphConv(in_channels, out_channels)
        self.tcn = TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x, A):
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.tcn(x)
        x = self.bn2(x)
        x = x + res
        x = self.relu(x)
        return x

class TrueSTGCN(nn.Module):
    def __init__(self, num_classes=4, in_channels=3):
        super().__init__()
        self.graph = Graph()
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        
        self.data_bn = nn.BatchNorm1d(in_channels * self.graph.num_node)
        
        self.st_gcn_networks = nn.ModuleList([
            STGCN_Block(in_channels, 64, kernel_size=9, stride=1, residual=False),
            STGCN_Block(64, 64, kernel_size=9, stride=1),
            STGCN_Block(64, 64, kernel_size=9, stride=1),
            STGCN_Block(64, 128, kernel_size=9, stride=2),
            STGCN_Block(128, 128, kernel_size=9, stride=1),
            STGCN_Block(128, 128, kernel_size=9, stride=1),
            STGCN_Block(128, 256, kernel_size=9, stride=2),
            STGCN_Block(256, 256, kernel_size=9, stride=1),
            STGCN_Block(256, 256, kernel_size=9, stride=1),
        ])
        
        self.fc = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous() # (N, M, V, C, T)
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)
        
        for gcn in self.st_gcn_networks:
            x = gcn(x, self.A)
            
        x = nn.functional.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1).mean(dim=1)
        
        x = self.dropout(x)
        x = self.fc(x)
        return x

#DATASET AND HETEROGENEOUS AUGMENTATION
#Mapping MediaPipe 33 joints to COCO 17 joints
MP_TO_COCO_MAPPING = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

import concurrent.futures

class RobustSkeletonDataset(Dataset):
    def __init__(self, metadata, target_frames=60, target_joints=17, is_training=False):
        self.metadata = metadata
        self.T = target_frames
        self.V = target_joints
        self.is_training = is_training
        
        print(f"\nPreloading {len(metadata)} samples into RAM for ultra-fast training...")
        self.cache_data = []
        self.cache_labels = []
        
        # We will use ThreadPoolExecutor to quickly read all tiny .npy files from disk
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            results = list(tqdm(executor.map(self._preload, metadata), total=len(metadata), desc="Caching Datasets"))
            
        for d, l in results:
            self.cache_data.append(d)
            self.cache_labels.append(l)

    def _preload(self, item):
        data = np.load(item["file_path"], allow_pickle=True).astype(np.float32)
        if data.ndim == 4 and data.shape[0] > 1: data = data[0]
        data = np.squeeze(data)
        
        if data.ndim == 2:
            t_len, feat_len = data.shape
            if feat_len == 34: data = data.reshape(t_len, 17, 2)
            elif feat_len == 51: data = data.reshape(t_len, 17, 3)
            else: data = data.reshape(t_len, feat_len // 2, 2)
                
        if data.ndim == 3:
            if data.shape[-1] in [2, 3]: data = np.transpose(data, (2, 0, 1))
            elif data.shape[1] in [2, 3]: data = np.transpose(data, (1, 0, 2))
                
        C, T_curr, V_curr = data.shape
        
        if V_curr == 33:
            data = data[:, :, MP_TO_COCO_MAPPING]
            V_curr = 17
            
        if T_curr != self.T:
            new_data = np.zeros((C, self.T, V_curr), dtype=np.float32)
            orig_indices = np.linspace(0, T_curr - 1, num=self.T)
            for c in range(C):
                for v in range(V_curr):
                    new_data[c, :, v] = np.interp(orig_indices, np.arange(T_curr), data[c, :, v])
            data = new_data
            T_curr = self.T
            
        if C == 2:
            z_channel = np.zeros((1, T_curr, V_curr), dtype=np.float32)
            data = np.concatenate([data, z_channel], axis=0)
            C = 3
            
        hip_center = (data[:, :, 11] + data[:, :, 12]) / 2.0
        data = data - hip_center[:, :, np.newaxis]
        
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val
            
        return data, item["label"]

    def __len__(self):
        return len(self.cache_data)

    def augment_skeleton(self, data):
        data = data.copy()
        if np.random.rand() > 0.5:
            data += np.random.normal(0, 0.02, data.shape).astype(np.float32)
        if np.random.rand() > 0.5 and data.shape[1] > 10:
            data = np.roll(data, np.random.randint(1, 10), axis=1)
        if np.random.rand() > 0.5:
            data *= np.random.uniform(0.9, 1.1)
        return data

    def __getitem__(self, idx):
        data = self.cache_data[idx]
        label = self.cache_labels[idx]
        
        if self.is_training: 
            data = self.augment_skeleton(data)

        out_data = np.zeros((3, self.T, self.V, 1), dtype=np.float32)
        c_limit = min(data.shape[0], 3)
        t_limit = min(data.shape[1], self.T)
        v_limit = min(data.shape[2], self.V)

        out_data[:c_limit, :t_limit, :v_limit, 0] = data[:c_limit, :t_limit, :v_limit]

        return torch.tensor(out_data), torch.tensor(label, dtype=torch.long)

#TRAINING PIPELINE
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

def extract_datasets():
    print(f"Extracting datasets from {ZIP_DIR} to {WORKING_DIR}...")
    zip_files = glob.glob(os.path.join(ZIP_DIR, "*.zip"))
    if not zip_files:
        print("No zip files found! Skipping extraction.")
        return
        
    for zip_file in zip_files:
        dataset_name = os.path.basename(zip_file).replace(".zip", "")
        extract_path = os.path.join(WORKING_DIR, dataset_name)
        os.makedirs(extract_path, exist_ok=True)
        print(f"Extracting {os.path.basename(zip_file)}...")
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
        except Exception as e:
            print(f"Error extracting {zip_file}: {e}")
            
def expand_bulk_files(x_path, y_path=None, label_map=None, default_label=None):
    """
    Expands X_train.npy into individual seq_X.npy files.
    If y_path is provided, applies label_map to true labels.
    Otherwise uses default_label.
    """
    data = np.load(x_path)
    if y_path:
        labels = np.load(y_path)
    else:
        labels = np.full((data.shape[0],), default_label)
        
    out_dir = os.path.join(os.path.dirname(x_path), "expanded")
    os.makedirs(out_dir, exist_ok=True)
    
    meta = []
    for i in range(data.shape[0]):
        lbl = int(labels[i])
        if label_map:
            if lbl not in label_map:
                continue
            lbl = label_map[lbl]
            
        sub_path = os.path.join(out_dir, f"seq_{i}.npy")
        if not os.path.exists(sub_path):
            np.save(sub_path, data[i])
        meta.append({"file_path": sub_path, "label": lbl})
    return meta

def map_classes():
    print("Mapping classes from unzipped folders...")
    metadata = []
    
    #Mapping ASDPose (Individual Files -> Stimming = 1)
    asdpose_files = glob.glob(os.path.join(WORKING_DIR, "asdpose_preprocessed", "**", "*.npy"), recursive=True)
    for f in tqdm(asdpose_files, desc="Parsing ASDPose"):
        metadata.append({"file_path": f, "label": 1})
        
    #Mapping SSBD (Bulk Files -> Stimming = 1)
    ssbd_x_train = glob.glob(os.path.join(WORKING_DIR, "ssbd_model_ready", "**", "X_train.npy"), recursive=True)
    ssbd_x_test = glob.glob(os.path.join(WORKING_DIR, "ssbd_model_ready", "**", "X_test.npy"), recursive=True)
    for f in ssbd_x_train + ssbd_x_test:
        metadata.extend(expand_bulk_files(f, default_label=1))
        
    #Mapping LE2I (Bulk Files -> 0: Normal, 1: Fall -> Mapped to 0: Normal, 2: Fall)
    le2i_x_train = glob.glob(os.path.join(WORKING_DIR, "le2i_preprocessed", "**", "X_train.npy"), recursive=True)
    le2i_x_test = glob.glob(os.path.join(WORKING_DIR, "le2i_preprocessed", "**", "X_test.npy"), recursive=True)
    for f in le2i_x_train + le2i_x_test:
        y_f = f.replace("X_", "y_")
        metadata.extend(expand_bulk_files(f, y_path=y_f if os.path.exists(y_f) else None, label_map={0: 0, 1: 2}, default_label=2))
        
    #Mapping HBD21 (Bulk Files -> Mapped to 3: Aggression, except 0 to 0)
    hbd_x_train = glob.glob(os.path.join(WORKING_DIR, "hbd21_test_train_split", "**", "X_train.npy"), recursive=True)
    hbd_x_test = glob.glob(os.path.join(WORKING_DIR, "hbd21_test_train_split", "**", "X_test.npy"), recursive=True)
    for f in hbd_x_train + hbd_x_test:
        y_f = f.replace("X_", "y_")
        metadata.extend(expand_bulk_files(f, y_path=y_f if os.path.exists(y_f) else None, label_map={0: 0, 1: 3, 2: 3, 3: 3}, default_label=3))
        
    return metadata

def train_model():
    #Prepare Data
    metadata = map_classes()
    if len(metadata) == 0:
        print("No valid .npy files found. Ensure extraction worked.")
        return
        
    print(f"Total samples for training pipeline: {len(metadata)}")
    
    #Stratified Split
    labels = [x["label"] for x in metadata]
    try:
        train_meta, val_meta = train_test_split(metadata, test_size=0.2, random_state=42, stratify=labels)
    except ValueError:
        print("Warning: Could not stratify. Falling back to random split.")
        train_meta, val_meta = train_test_split(metadata, test_size=0.2, random_state=42)

    #Class Balancing using WeightedRandomSampler
    train_labels = [x["label"] for x in train_meta]
    class_counts = np.bincount(train_labels, minlength=4)
    print(f"Train Class Distribution: {class_counts}")
    
    class_weights = 1.0 / (class_counts + 1e-6) # Avoid division by zero
    sample_weights = [class_weights[label] for label in train_labels]
    
    # We cap num_samples to 4000 per epoch to prevent massive overfitting on the tiny Fall dataset 
    # (which only has 144 samples) and to make the epochs blisteringly fast.
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=4000, replacement=True)

    train_loader = DataLoader(RobustSkeletonDataset(train_meta, is_training=True), batch_size=256, sampler=sampler, num_workers=4 if RUNNING_ON_COLAB else 0, pin_memory=True)
    val_loader = DataLoader(RobustSkeletonDataset(val_meta, is_training=False), batch_size=256, shuffle=False, num_workers=4 if RUNNING_ON_COLAB else 0, pin_memory=True)

    #Setup Model & Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrueSTGCN(num_classes=4).to(device)
    
    criterion = LabelSmoothingLoss(classes=4, smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    EPOCHS = 50
    best_val_acc = 0.0
    patience, max_patience = 0, 12
    MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_true_stgcn.pth")

    print(f"\nStarting True ST-GCN Training on {device}")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loop.set_postfix(loss=loss.item())

        train_acc = 100 * correct / total
        scheduler.step()

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        if val_total > 0:
            val_acc = 100 * val_correct / val_total
        else:
            val_acc = 0.0
            
        print(f"-> Train Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            patience = 0
            print("Model Saved!")
        else:
            patience += 1
            
        if patience >= max_patience:
            print(f"\nEarly stopping at epoch {epoch+1} (No improvement for {max_patience} epochs)")
            break

    print(f"\n--- Training Complete! Best Val Accuracy: {best_val_acc:.2f}% ---")
    print(f"Model saved to: {MODEL_SAVE_PATH}")

# --- MULTIMODAL AUDIO INTEGRATION ---

class AudioFeatureExtractor:
    def __init__(self, sample_rate=22050, n_mfcc=13):
        self.sr = sample_rate
        self.n_mfcc = n_mfcc

    def extract_features(self, audio_path):
        """Extracts RMS energy (volume) and MFCCs from an audio file."""
        if 'librosa' not in globals():
            raise ImportError("Please install librosa: pip install librosa")
            
        y, sr = librosa.load(audio_path, sr=self.sr)
        rms_energy = librosa.feature.rms(y=y)[0]
        max_energy = np.max(rms_energy)
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        return max_energy, mfccs_mean

class AudioClassifier(nn.Module):
    """Lightweight classifier for MFCC sound signatures."""
    def __init__(self, input_dim=13, num_classes=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        return self.fc(x)

class YOLOFallDetector:
    def __init__(self, model_path="yolov8n.pt"):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.available = True
        except ImportError:
            self.available = False

    def detect_fall(self, video_frames):
        """
        Returns 1.0 if a person's bounding box is horizontal (width > height).
        """
        if not self.available or len(video_frames) == 0:
            return 0.0
            
        max_aspect_ratio = 0.0
        
        for frame in video_frames:
            results = self.model(frame, verbose=False)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.7: # 0 is 'person'
                        x1, y1, x2, y2 = box.xyxy[0]
                        width = float(x2 - x1)
                        height = float(y2 - y1)
                        if height > 0:
                            aspect_ratio = width / height
                            if aspect_ratio > max_aspect_ratio:
                                max_aspect_ratio = aspect_ratio
                                
        if max_aspect_ratio >= 1.2:
            return 1.0
        return 0.0

class MultimodalFusionSystem:
    def __init__(self, stgcn_model, audio_model=None, volume_threshold=0.5, enable_yolo=True):
        self.stgcn = stgcn_model
        self.audio_model = audio_model
        self.volume_threshold = volume_threshold
        self.yolo = YOLOFallDetector() if enable_yolo else None
        
    def calculate_brightness(self, video_frames):
        """Calculates average brightness of video frames to detect low-light."""
        import cv2
        brightness_vals = [np.mean(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) for f in video_frames]
        return np.mean(brightness_vals) if brightness_vals else 255.0

    def infer(self, skeleton_data, video_frames, audio_features, device='cpu'):
        """
        Late Fusion: Combines visual score and audio score dynamically based on lighting.
        """
        self.stgcn.eval()
        with torch.no_grad():
            # 1. Visual Score (STGCN)
            vis_logits = self.stgcn(skeleton_data.to(device))
            vis_probs = torch.softmax(vis_logits, dim=1).cpu().numpy()[0]
            
            # --- YOLO ENSEMBLE OVERRIDE ---
            yolo_triggered = False
            if self.yolo is not None and self.yolo.available:
                if self.yolo.detect_fall(video_frames) == 1.0:
                    vis_probs = np.array([0.0, 0.0, 1.0, 0.0]) # Force Fall class visually
                    yolo_triggered = True
            
            # 2. Audio Score
            audio_probs = np.zeros_like(vis_probs)
            max_energy, mfccs_mean = audio_features
            
            if self.audio_model is not None:
                self.audio_model.eval()
                mfcc_tensor = torch.tensor(mfccs_mean, dtype=torch.float32).unsqueeze(0).to(device)
                aud_logits = self.audio_model(mfcc_tensor)
                audio_probs = torch.softmax(aud_logits, dim=1).cpu().numpy()[0]
            else:
                # Fallback to volume thresholding (assuming class 2 is Anomaly/Fall)
                if max_energy > self.volume_threshold:
                    audio_probs[2] = 1.0 
                else:
                    audio_probs[0] = 1.0 

            # 3. Dynamic Weighting based on Brightness
            brightness = self.calculate_brightness(video_frames)
            
            # CRITICAL OVERRIDE: If the sound is exceptionally loud (e.g., a crash), 
            # we should heavily trust the audio even in broad daylight, because the 
            # camera model might have missed the fall (e.g., occlusion or bad angle).
            if max_energy > (self.volume_threshold * 2.0):
                vis_weight, aud_weight = 0.3, 0.7
            elif brightness < 30:
                vis_weight, aud_weight = 0.2, 0.8
            elif brightness < 80:
                vis_weight, aud_weight = 0.5, 0.5
            else:
                vis_weight, aud_weight = 0.8, 0.2
                
            final_probs = (vis_weight * vis_probs) + (aud_weight * audio_probs)
            final_pred = np.argmax(final_probs)
            
            return final_pred, final_probs, yolo_triggered

if __name__ == "__main__":
    extract_datasets()
    train_model()
