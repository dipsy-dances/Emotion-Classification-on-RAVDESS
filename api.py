"""
FastAPI Inference Server — Speech Emotion Recognition
Run: uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import torch
import torch.nn as nn
import numpy as np
import librosa
import json
import io
import os
import tempfile
from pathlib import Path

app = FastAPI(title="Speech Emotion Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve frontend ──────────────────────────────────────────
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_frontend():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"message": "API running. POST /predict with a .wav file."}


# ── Model definition (must match train.py) ──────────────────

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class EmotionCNN(nn.Module):
    def __init__(self, n_classes=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            ResBlock(32), nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            ResBlock(64), nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            ResBlock(128), nn.MaxPool2d(2, 2), nn.Dropout2d(0.3),
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.encoder(x))


# ── Load model at startup ────────────────────────────────────

MODEL_PATH = Path(os.getenv("MODEL_PATH", "./checkpoints/best_model.pt"))
model      = None
emotions   = None
cfg        = None


def load_model():
    global model, emotions, cfg
    if not MODEL_PATH.exists():
        print(f"⚠  No model found at {MODEL_PATH}. Serving demo mode.")
        return

    ckpt     = torch.load(MODEL_PATH, map_location="cpu")
    emotions = ckpt["emotions"]
    model    = EmotionCNN(n_classes=len(emotions))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    cfg_path = MODEL_PATH.parent / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
    print(f"✓ Model loaded. Val acc: {ckpt.get('val_acc', '?'):.3f}")


@app.on_event("startup")
def startup():
    load_model()


# ── Feature extraction (mirrors train.py) ───────────────────

def extract_features(audio_bytes: bytes) -> torch.Tensor:
    SR         = cfg["sr"]         if cfg else 22050
    DURATION   = cfg["duration"]   if cfg else 3.0
    N_MELS     = cfg["n_mels"]     if cfg else 128
    N_FFT      = cfg["n_fft"]      if cfg else 2048
    HOP_LENGTH = cfg["hop_length"] if cfg else 512
    MAX_LEN    = cfg["max_len"]    if cfg else 130

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        y, _ = librosa.load(tmp_path, sr=SR, duration=DURATION)
    finally:
        os.unlink(tmp_path)

    target = int(SR * DURATION)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        y = y[:target]

    mel     = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS,
                                              n_fft=N_FFT, hop_length=HOP_LENGTH)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)

    if log_mel.shape[1] < MAX_LEN:
        log_mel = np.pad(log_mel, ((0, 0), (0, MAX_LEN - log_mel.shape[1])))
    else:
        log_mel = log_mel[:, :MAX_LEN]

    return torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


# ── Predict endpoint ─────────────────────────────────────────

EMOTION_EMOJI = {
    "neutral": "😐", "calm": "😌", "happy": "😊", "sad": "😢",
    "angry": "😠", "fearful": "😨", "disgust": "🤢", "surprised": "😲"
}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(503, "Model not loaded. Run train.py first.")
    if not file.filename.endswith((".wav", ".mp3", ".ogg", ".flac")):
        raise HTTPException(400, "Audio file required (.wav, .mp3, .ogg, .flac)")

    audio_bytes = await file.read()
    if len(audio_bytes) > 10 * 1024 * 1024:
        raise HTTPException(413, "File too large (max 10 MB)")

    try:
        x = extract_features(audio_bytes)
    except Exception as e:
        raise HTTPException(422, f"Could not process audio: {e}")

    with torch.no_grad():
        logits = model(x).squeeze()
        probs  = torch.softmax(logits, dim=-1).numpy()

    top_idx    = int(np.argmax(probs))
    prediction = emotions[top_idx]
    confidence = float(probs[top_idx])

    all_probs = {
        emotions[i]: {
            "probability": float(probs[i]),
            "percentage":  round(float(probs[i]) * 100, 1),
            "emoji":       EMOTION_EMOJI.get(emotions[i], "")
        }
        for i in range(len(emotions))
    }

    return {
        "prediction":  prediction,
        "confidence":  round(confidence * 100, 1),
        "emoji":       EMOTION_EMOJI.get(prediction, ""),
        "all_emotions": all_probs,
        "model_info":  {"val_acc": "see checkpoints/"}
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
