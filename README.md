# 🎙️ Speech Emotion Recognition

Deep learning–based Speech Emotion Recognition (SER) system built with PyTorch and FastAPI using the RAVDESS dataset.

The project extracts Mel Spectrogram features from speech audio and classifies emotions using a custom CNN with Residual Blocks.

---

##  Features

- Speech Emotion Recognition from audio files
- CNN + Residual Network architecture
- Mel Spectrogram feature extraction
- FastAPI inference server
- Real-time emotion prediction API
- Training visualizations (confusion matrix & learning curves)
- Audio augmentation for improved generalization

---

##  Supported Emotions

- Neutral 
- Calm 
- Happy 
- Sad 
- Angry 
- Fearful 
- Disgust 
- Surprised 

---

## 📂 Project Structure

```bash
.
├── train.py
├── api.py
├── checkpoints/
├── static/
└── README.md
