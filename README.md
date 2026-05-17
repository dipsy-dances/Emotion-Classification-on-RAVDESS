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

## Training Details
- **Augmentation:** Time shift, SpecAugment (freq + time masking)
- **Loss:** CrossEntropy with label smoothing (0.1)
- **Optimizer:** AdamW (lr=1e-3, weight_decay=1e-4)
- **Scheduler:** Cosine Annealing
- **Batch size:** 32
- **Epochs:** 50

---

## Demo Mode (no trained model needed)
Open `index.html` directly in browser — it uses the **Claude API** to predict
emotion from browser-extracted audio features (RMS energy, ZCR, spectral centroid).
Switch `DEMO_MODE = false` in the JS once your model is trained.

---

## API Endpoints
```
POST /predict     — Upload audio, get emotion prediction
GET  /health      — Check if model is loaded
GET  /            — Serves index.html
```

### Example curl
```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@your_audio.wav" | python -m json.tool
```

### Response
```json
{
  "prediction": "happy",
  "confidence": 74.2,
  "emoji": "😊",
  "all_emotions": {
    "happy": {"probability": 0.742, "percentage": 74.2, "emoji": "😊"},
    ...
  }
}
```

---

## Deployment (HuggingFace Spaces — Free)
1. Create a new Space on huggingface.co (Gradio or Docker SDK)
2. Upload: `train.py`, `api.py`, `index.html`, `requirements.txt`, `best_model.pt`, `config.json`
3. For Gradio: wrap predict() in a gr.Interface
4. For Docker: use the FastAPI app directly

This project is a **resume-ready deployed ML app** — add the HuggingFace Spaces link to your resume.

---

## Put This on Your Resume As:
> **Speech Emotion Recognition Web App** | Python, PyTorch, FastAPI, librosa  
> Built CNN (ResBlock architecture) on RAVDESS dataset achieving ~70% accuracy across 8 emotion classes. Deployed as a full-stack web app with live audio recording, mel-spectrogram feature extraction, and real-time inference via FastAPI REST API. Hosted on HuggingFace Spaces.
