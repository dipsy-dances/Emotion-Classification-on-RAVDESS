# Speech Emotion Recognition — VoiceRead
**RAVDESS Dataset · CNN on Mel Spectrograms · FastAPI + Web App**

---

## Project Structure
```
├── train.py          ← Full training pipeline (run this first)
├── api.py            ← FastAPI inference server
├── index.html        ← Web frontend (copy to static/ for FastAPI to serve)
├── requirements.txt  ← Python dependencies
└── checkpoints/      ← Created after training
    ├── best_model.pt
    └── config.json
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download dataset
```bash
pip install kaggle
kaggle datasets download -d uwrfkaggler/ravdess-emotional-speech-audio
unzip ravdess-emotional-speech-audio.zip -d ./audio_speech_actors_01-24
```

### 3. Train the model
```bash
python train.py --data_dir ./audio_speech_actors_01-24 --output_dir ./checkpoints
# Expected: ~65-72% val accuracy after 50 epochs on CPU; higher with GPU
```

### 4. Run the API
```bash
mkdir -p static && cp index.html static/
uvicorn api:app --reload --port 8000
# Open http://localhost:8000
```

---

## Model Architecture
```
Input: Log-Mel Spectrogram (1 × 128 × 130)
  ↓
Conv Block 1: Conv2d(1→32) + ResBlock + MaxPool + Dropout
Conv Block 2: Conv2d(32→64) + ResBlock + MaxPool + Dropout  
Conv Block 3: Conv2d(64→128) + ResBlock + MaxPool + Dropout
Conv Block 4: Conv2d(128→256) + AdaptiveAvgPool(4×4)
  ↓
Flatten → FC(4096→512) → ReLU → Dropout
         → FC(512→128) → ReLU → Dropout
         → FC(128→8)
  ↓
Output: 8-class softmax (neutral/calm/happy/sad/angry/fearful/disgust/surprised)
```

