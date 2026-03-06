# Spoken Digit Recognition with LSTM

Classifying spoken digits (0–9) from raw audio using LSTM networks. Four experiments compare raw waveforms vs. mel spectrograms, with and without data augmentation — showing how feature engineering dramatically impacts model performance.

---

## Results Summary

| # | Model | Train Loss | Val Loss | Val F1 Score |
|---|-------|-----------|----------|-------------|
| 1 | Raw Waveform + LSTM | 2.3028 | 2.3026 | 0.100 |
| 2 | Mel Spectrogram + LSTM | 0.2740 | 0.2633 | **0.917** |
| 3 | Augmented Raw Waveform + LSTM | 2.3028 | 2.3026 | 0.100 |
| 4 | Augmented Mel Spectrogram + LSTM | 0.0642 | 0.1164 | **0.955** |

**Key takeaway:** Raw waveforms fed directly into an LSTM fail to learn meaningful patterns (F1 ≈ 0.10, random chance for 10 classes). Converting to mel spectrograms gives a massive jump to 91.7%, and combining spectrograms with data augmentation achieves **95.5% F1**.

---

## Dataset

[Free Spoken Digit Dataset (FSDD)](https://github.com/Jakobovski/free-spoken-digit-dataset)

- 2,000 WAV recordings (digits 0–9)
- Multiple speakers: `jackson`, `theo`, `nicolas`, `yweweler`
- Each speaker recorded each digit ~50 times
- Sampling rate: 22,050 Hz
- File naming: `{digit}_{speaker}_{index}.wav` (e.g. `0_jackson_0.wav`)

The dataset is not included in this repo. Download `recordings.zip` from the FSDD link above.

---

## Project Structure

```
.
├── speech-detection-assignment.pdf   # Full Colab notebook with code and outputs
├── requirements.txt
└── README.md
```

---

## Approach

### Preprocessing

All audio is loaded at 22,050 Hz. Duration analysis showed 99% of clips fall under 0.8 seconds, so sequences are padded/truncated to **17,640 samples** (0.8 × 22,050). A boolean masking vector is created alongside to tell the LSTM which timesteps are real vs. padding.

```
max_length = 17640   # 0.8s * 22050 Hz
```

### Feature Representations

**Raw waveform** — the raw sample array, padded to 17,640 timesteps. Input shape: `(17640, 1)`.

**Mel spectrogram** — each raw array is converted using Librosa:
```python
spectrum = librosa.feature.melspectrogram(y=raw, sr=22050, n_mels=64)
log_mel  = librosa.power_to_db(spectrum, ref=np.max)
# Output shape: (64, 35)  — 64 mel bands × 35 time frames
```

### Data Augmentation

Applied only to training data, generating **9 augmented copies** per original clip:
- **Time stretching**: 0.7×, 1.0×, 1.3× speed (−30% / no change / +30%)
- **Pitch shifting**: −1, 0, +1 half-steps

This expands 1,600 training samples to **14,400**.

---

## Model Architectures

### Model 1 & 3 — Raw Waveform LSTM

```
Input     (None, 17640, 1)   raw waveform
Mask      (None, 17640)      boolean mask
LSTM      25 units           with masking
Dense     50, ReLU
Dense     10, Softmax        output (digit 0–9)

Total params: 4,510
```

### Model 2 & 4 — Mel Spectrogram LSTM

```
Input     (None, 64, 35)     log-mel spectrogram
LSTM      256, return_sequences=True   → (None, 64, 256)
ReduceMean over last axis              → (None, 64)
Dense     256, ReLU
Dense     128, ReLU
Dense     10,  Softmax

Total params: 349,834
```

The `reduce_mean` over the LSTM output timesteps acts as a global average pool, summarizing the temporal features before the dense classifier.

---

## Training Details

- **Optimizer**: Adam (lr=0.001)
- **Loss**: Sparse categorical cross-entropy
- **Metric**: Micro-averaged F1 score (custom Keras callback)
- **Callbacks**: EarlyStopping + TensorBoard (with gradient histograms)
- **Split**: 70/30 (Models 1 & 2) · 80/20 (Models 3 & 4)

---

## Setup & Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get the dataset

```bash
# Download and extract FSDD recordings
# https://github.com/Jakobovski/free-spoken-digit-dataset
# Place the .wav files in: ./recordings/
```

### 3. Run

Open `speech-detection-assignment.pdf` to follow the full notebook workflow, or adapt the code cells into a `.ipynb` / `.py` file.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `librosa` | Audio loading, mel spectrograms, augmentation |
| `tensorflow` | LSTM models, training |
| `numpy` | Numerical arrays, padding |
| `pandas` | DataFrame management |
| `scikit-learn` | Train/test split, F1 score |
| `matplotlib` / `seaborn` | Visualizations |
| `prettytable` | Results table formatting |

---

## Analysis & Conclusions

1. **Raw waveforms are too noisy for LSTM to learn from directly.** The model stuck at chance-level performance (F1 = 0.10) regardless of augmentation, because the temporal patterns in raw audio at 22 kHz are too fine-grained for the LSTM to extract meaningful digit-level features.

2. **Mel spectrograms encode perceptually relevant features.** By compressing the time-frequency content into 64 mel bands × 35 frames, the representation is compact enough for the LSTM to identify patterns associated with each digit. F1 jumped from 0.10 → 0.917.

3. **Data augmentation further reduces overfitting.** Adding 9× augmentation via time-stretch and pitch-shift increased training data from 1,600 → 14,400 samples, pushing Val F1 from 0.917 → **0.955** and reducing train loss from 0.274 → 0.064.

4. **Architecture insight:** The spectrogram model uses `return_sequences=True` with a subsequent `reduce_mean`, which is effectively a temporal attention-free pooling mechanism. This outperforms just taking the final LSTM hidden state because it aggregates information across all time steps.
