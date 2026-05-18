# EEG-to-Speech: Multi-Subject Acoustic Decoder

> Decoding imagined and perceived speech directly from raw EEG brain signals — 
> building a neural architecture that maps electrical brain activity to audible speech.

---

## Overview

This project trains a **single neural network to decode speech acoustics from EEG brain signals across 15 subjects simultaneously**, despite every subject having a different number of EEG channels. The model predicts mel spectrograms from raw EEG, then synthesises speech audio using either Griffin-Lim or the HiFi-GAN neural vocoder.

This is **Milestone 1** of a larger EEG-to-speech system. The goal here is to establish a strong per-subject and cross-subject acoustic baseline before fusing additional modalities.

---

## Key Results

| Metric | Previous Baseline (1 subject) | **This Work (15 subjects)** | Improvement |
|---|---|---|---|
| Pearson Correlation | 0.361 | **0.968** | **+168%** |
| Cosine Similarity | 0.923 | **0.990** | +7% |
| Spectral Convergence | 0.415 | **0.185** | −55% error |
| Subjects Covered | 1 / 15 | **15 / 15** | Complete |

**Per-subject Pearson correlations** (all 15 subjects, full-recording reconstruction):

| Subject | Channels | Pearson r |
|---|---|---|
| sub-01 | 127 | 0.9699 |
| sub-02 | 127 | 0.9766 |
| sub-03 | 127 | 0.9778 |
| sub-04 | 115 | 0.9774 |
| sub-05 | 60  | 0.9701 |
| sub-06 | 127 | 0.9752 |
| sub-07 | 127 | 0.9751 |
| sub-08 | 54  | 0.9742 |
| sub-09 | 117 | 0.9708 |
| sub-10 | 122 | 0.9517 |
| sub-11 | 68  | 0.9751 |
| sub-12 | 127 | 0.9733 |
| sub-13 | 157 | 0.9535 |
| sub-14 | 127 | 0.9552 |
| sub-15 | 65  | 0.9486 |
| **Mean** | — | **0.968** |

---

## Technical Novelties

### 1. Variable-Channel Multi-Subject Training
Different EEG recording setups capture different numbers of electrodes per participant (54 to 157 channels across 15 subjects). Most prior work trains a **separate model per subject**. Here, a single model handles all subjects simultaneously via **subject-specific input projectors** — one linear layer per subject that maps that subject's variable-width EEG feature vector to a shared 256-dimensional space. This means the shared Transformer backbone sees a uniform representation regardless of who the subject is, enabling genuine multi-subject transfer learning.

### 2. Transformer Encoder–Decoder with Cross-Attention
The architecture is a full **seq2seq Transformer** designed specifically for EEG-to-mel transduction:

- **EEG Encoder** — 4-layer Transformer encoder with 8 attention heads reads the projected EEG sequence and builds a rich contextual representation where every time-frame attends to every other frame.
- **Mel Decoder** — 4-layer autoregressive Transformer decoder generates mel frames using:
  - **Causal self-attention** (frame *t* only attends to frames 0…*t*−1)
  - **Cross-attention** to the full EEG encoder output at every layer
- **Prenet** — Two-layer MLP with always-on dropout (even at inference) applied to the mel input, preventing the decoder from trivially copying the previous frame and forcing it to rely on the EEG cross-attention signal.

### 3. PostNet Residual Refinement
After the decoder produces a raw mel prediction, a **5-layer 1D convolutional PostNet** (512 channels, kernel size 5, BatchNorm + Tanh) adds a learned residual correction:

```
mel_final = mel_raw + postnet(mel_raw)
```

This two-stage prediction — coarse from the Transformer, fine-grained from the CNN — is inspired by Tacotron 2 and substantially reduces spectral convergence error.

### 4. Overlap-Add Full-Recording Reconstruction
Training uses 512-frame windows with 50% overlap. For evaluation and audio generation, a **sliding overlap-add scheme** reconstructs the entire ~5-minute recording per subject by averaging overlapping predictions. This avoids boundary artefacts and produces consistent full-length mel spectrograms.

### 5. Dual Vocoder Comparison: Griffin-Lim vs. HiFi-GAN
Two vocoders are benchmarked to convert predicted mel spectrograms to audio:

- **Griffin-Lim** (iterative phase estimation, 64 iterations) — fast, no neural parameters, noisier output
- **HiFi-GAN Universal Vocoder** — a pretrained GAN-based neural vocoder producing significantly higher audio quality; integrated via a custom percentile-based mel normalisation and bilinear upsampling (23 → 80 mel bins) to bridge the dataset's feature space to the vocoder's expected input

---

## Architecture Summary

```
EEG input  (T, flat_dim)         flat_dim = n_channels × 9 lags
     │
     ▼
SubjectProjector_i               Linear(flat_dim_i → 256)   [per-subject]
     │
     ▼
PositionalEncoding
     │
     ▼
TransformerEncoder               4 layers · 8 heads · d_model=256 · ff=1024
     │  memory
     ▼
Prenet(mel_shifted) ──────────── 2× Linear + Dropout(0.5)   [always on]
     │
     ▼
TransformerDecoder               4 layers · 8 heads · causal mask + cross-attention
     │
     ▼
Linear(256 → 23)   →  mel_raw
     │
     ▼
PostNet (5× Conv1d-BN-Tanh)  →  residual correction
     │
     ▼
mel_final  (T, 23)
     │
     ▼
Griffin-Lim  or  HiFi-GAN  →  audio waveform
```

---

## Dataset

- **15 subjects**, publicly available EEG speech perception dataset
- **Variable EEG channels**: 54 – 157 electrodes per subject
- **EEG features**: flattened 9-lag temporal context per channel (`flat_dim = n_channels × 9`)
- **Target**: 23-bin log mel spectrogram at 100 fps (10 ms frame shift)
- **Windowing**: 512-frame windows (5.12 s), 50% stride → ~1,480 training / ~254 validation windows
- **Split**: 85% train / 15% validation, seeded for reproducibility (`seed=42`)

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Epochs | 50 |
| Optimiser | Adam (lr = 1e-4) |
| Scheduler | ReduceLROnPlateau (factor 0.5, patience 5) |
| Loss | L1 (Mean Absolute Error) |
| Gradient clipping | norm ≤ 1.0 |
| Batch size | 1 window (heterogeneous channel widths prevent stacking) |
| Best epoch | 22 (val loss = 0.4865) |

**L1 loss** was chosen over MSE because mel spectrograms contain occasional large energy spikes; L1 is more robust to outliers as it does not square the residual.

---

## Technology Stack

- **PyTorch** — model, training loop, DataLoader
- **NumPy / SciPy** — preprocessing, metrics (Pearson, cosine similarity, spectral convergence)
- **librosa** — Griffin-Lim vocoding, mel filterbank
- **HiFi-GAN** (jik876) — neural vocoder
- **Matplotlib** — spectrogram visualisations, training curves, per-subject analysis plots
- **soundfile** — audio export

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Pearson r** | Linear correlation between predicted and ground-truth mel frames |
| **Cosine Similarity** | Direction alignment in high-dimensional mel space |
| **Spectral Convergence** | Normalised Frobenius norm of the prediction error |
| **MAE** | Mean absolute error on mel values |

Metrics are computed both on held-out **validation windows** and on the **full-recording overlap-add reconstruction** to confirm that performance holds across the full continuous signal.

---

## File Structure

```
├── Milestone_1_Acoustic_Decoder_HiFiGAN.ipynb   # Full pipeline notebook
└── README.md
```

---

## What's Next (Milestone 2)

This acoustic-only decoder establishes the baseline. The next milestone integrates additional modalities (semantic / phonetic features) into a **trimodal model**, and compares full-recording Pearson and spectral convergence against the numbers reported here.
