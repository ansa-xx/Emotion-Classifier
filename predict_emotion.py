import os
import sys
import torch
import torch.nn as nn
from whisper.audio import load_audio, log_mel_spectrogram

# Add FFmpeg path if needed
os.environ["PATH"] += os.pathsep + "add path to ffmpeg"

# Label mapping (must match training!)
label_map = {
    0: 'Angry',
    1: 'Happy',
    2: 'Neutral',
    3: 'Sad'
}

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CNN model (same as used in training)
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
            nn.Linear(256, len(label_map))
        )

    def forward(self, x):
        return self.fc(self.net(x))

# Load trained model
model = EnhancedEmotionCNN().to(device)
model.load_state_dict(torch.load("final_emotion_cnn.pth", map_location=device))
model.eval()

def predict_emotion(filepath):
    audio = load_audio(filepath)
    mel = log_mel_spectrogram(audio)
    mel = mel.unsqueeze(0).unsqueeze(0)  # [1, 1, freq, time]
    mel = torch.nn.functional.interpolate(mel, size=(128, 128), mode='bilinear', align_corners=False)
    mel = mel.squeeze(0).repeat(3, 1, 1)  # [3, 128, 128]
    mel = mel.unsqueeze(0).to(device)    # [1, 3, 128, 128]

    with torch.no_grad():
        outputs = model(mel)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

    return label_map[pred_idx], confidence

# CLI usage
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_emotion.py path/to/audio.wav")
        sys.exit(1)

    audio_path = sys.argv[1]
    if not os.path.isfile(audio_path):
        print(f" File not found: {audio_path}")
        sys.exit(1)

    emotion, confidence = predict_emotion(audio_path)
    print(f" Predicted Emotion: {emotion} ({confidence * 100:.2f}% confidence)")
