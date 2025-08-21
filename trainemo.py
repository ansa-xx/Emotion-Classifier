import os
os.environ["PATH"] += os.pathsep + "/home/ec2-user/project/ffmpeg/ffmpeg-git-20240629-amd64-static"

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from whisper.audio import load_audio, log_mel_spectrogram
from tqdm import tqdm
from collections import Counter
import random
from sklearn.metrics import classification_report
import numpy as np

# Reproducibility
seed = 42
torch.manual_seed(seed)
random.seed(seed)

# Emotion labels
label_map = {
    'Angry': 0,
    'Happy': 1,
    'Neutral': 2,
    'Sad': 3
}
num_classes = len(label_map)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
class EmotionDataset(Dataset):
    def __init__(self, root_dir, label_map):
        self.filepaths = []
        self.labels = []
        for label_name in label_map:
            folder = os.path.join(root_dir, label_name)
            if os.path.isdir(folder):
                for f in os.listdir(folder):
                    if f.endswith(".wav"):
                        self.filepaths.append(os.path.join(folder, f))
                        self.labels.append(label_map[label_name])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        audio_path = self.filepaths[idx]
        label = self.labels[idx]
        audio = load_audio(audio_path)
        mel = log_mel_spectrogram(audio)
        mel = mel.unsqueeze(0).unsqueeze(0)
        mel = torch.nn.functional.interpolate(mel, size=(128, 128), mode='bilinear', align_corners=False)
        mel = mel.squeeze(0).repeat(3, 1, 1)
        return mel, label

# Enhanced CNN model
class EnhancedEmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.fc(self.net(x))

# Class weight computation
def compute_class_weights(labels):
    counts = Counter(labels)
    total = sum(counts.values())
    return torch.tensor([total / counts[i] for i in range(num_classes)], dtype=torch.float)

# Load data
data_dir = "combined_dataset"
dataset = EmotionDataset(data_dir, label_map)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# Model setup
model = EnhancedEmotionCNN().to(device)
weights = compute_class_weights(dataset.labels).to(device)
loss_fn = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

# Training loop
num_epochs = 15
best_val_acc = 0

for epoch in range(num_epochs):
    model.train()
    correct, total, running_loss = 0, 0, 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        running_loss += loss.item()
        loop.set_postfix(loss=running_loss / total, acc=correct / total)

    # Validation
    model.eval()
    val_correct = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1)
            val_correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = val_correct / len(val_dataset)
    scheduler.step(val_acc)
    best_val_acc = max(best_val_acc, val_acc)
    print(f"Validation Accuracy: {val_acc:.4f}")

# Save model
torch.save(model.state_dict(), "final_emotion_cnn.pth")
print(" Training complete.")
print(f" Best validation accuracy: {best_val_acc:.4f}")
print(" Model saved as 'final_emotion_cnn.pth'")

# Text-based classification report
print("\n Classification Report:")
print(classification_report(all_labels, all_preds, target_names=label_map.keys()))
