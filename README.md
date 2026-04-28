# 🎙️ End-to-End Speech Understanding Pipeline for Code-Switched Audio

<p align="center">
  <b>A Robust Multistage Deep Learning System for Hinglish Speech Processing</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg"/>
  <img src="https://img.shields.io/badge/PyTorch-DeepLearning-red.svg"/>
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg"/>
  <img src="https://img.shields.io/badge/ASR-Whisper-green.svg"/>
  <img src="https://img.shields.io/badge/LID-wav2vec2-purple.svg"/>
  <img src="https://img.shields.io/badge/SpeechBrain-Embeddings-orange.svg"/>
  <img src="https://img.shields.io/badge/Adversarial-FGSM-critical.svg"/>
  <img src="https://img.shields.io/badge/Status-Complete-success.svg"/>
</p>

---

## 📌 Overview

This repository implements a **comprehensive, end-to-end speech understanding pipeline** designed for **real-world noisy and code-switched (Hinglish) audio**.

The system integrates multiple state-of-the-art deep learning components to transform raw speech into **structured linguistic and semantic representations**, while also evaluating **robustness under adversarial perturbations**.

---

## 🚀 Key Highlights

- ✔️ Handles **noisy real-world audio**
- ✔️ Supports **code-switched speech (Hindi + English)**
- ✔️ Combines **speech, NLP, and speaker recognition**
- ✔️ Includes **adversarial robustness testing (FGSM)**
- ✔️ Modular pipeline — each stage independently analyzable

---

## 🧠 Pipeline Architecture

```text
                ┌────────────────────────┐
                │      Raw Audio         │
                └──────────┬─────────────┘
                           │
                           ▼
                ┌────────────────────────┐
                │   Speech Denoising     │
                └──────────┬─────────────┘
                           │
                           ▼
                ┌────────────────────────┐
                │   ASR (Whisper Model)  │
                └──────────┬─────────────┘
                           │
                           ▼
                ┌────────────────────────┐
                │ Language Identification│
                │     (wav2vec2)         │
                └──────────┬─────────────┘
                           │
                           ▼
                ┌────────────────────────┐
                │   Phonetic Mapping     │
                │        (IPA)           │
                └──────────┬─────────────┘
                           │
                           ▼
                ┌────────────────────────┐
                │  Semantic Translation  │
                │ Hinglish → Maithili    │
                └──────────┬─────────────┘
                           │
                           ▼
                ┌────────────────────────┐
                │ Speaker Embeddings     │
                │     (SpeechBrain)      │
                └──────────┬─────────────┘
                           │
                           ▼
                ┌────────────────────────┐
                │ Adversarial Attack     │
                │        (FGSM)          │
                └────────────────────────┘
````

---

## 🧩 Core Modules

### 🔊 1. Speech Denoising

* Enhances audio quality for downstream tasks
* Handles real-world noise conditions

---

### 🗣️ 2. Automatic Speech Recognition (ASR)

* Based on **Whisper**
* Produces high-quality transcriptions
* Handles mixed-language speech effectively

---

### 🌍 3. Language Identification (LID)

* Frame-level classification using **wav2vec2**
* Detects **Hindi vs English switching**
* Enables fine-grained linguistic analysis

---

### 🔤 4. Phonetic Mapping (IPA)

* Converts text → phoneme representation
* Useful for linguistic and pronunciation modeling

---

### 🌐 5. Semantic Translation

* Hinglish → Maithili translation
* Transformer-based multilingual modeling

---

### 🧑‍🎤 6. Speaker Embedding Extraction

* Uses **SpeechBrain ECAPA-TDNN**
* Captures speaker identity features
* Useful for speaker verification & clustering

---

### ⚔️ 7. Adversarial Robustness (FGSM)

* Applies **Fast Gradient Sign Method**
* Tests system vulnerability to perturbations
* Demonstrates robustness limitations

---

## 🛠️ Tech Stack

| Category         | Tools                    |
| ---------------- | ------------------------ |
| Deep Learning    | PyTorch                  |
| Audio Processing | Librosa, Torchaudio      |
| ASR              | OpenAI Whisper           |
| LID              | wav2vec 2.0              |
| NLP              | HuggingFace Transformers |
| Speaker Models   | SpeechBrain              |
| Phonetics        | Epitran, Phonemizer      |

---

## 📂 Repository Structure

```bash
.
├── notebook.ipynb            # Full pipeline implementation
├── audio/                    # Input audio samples
├── outputs/                  # Generated outputs
│   ├── transcriptions/
│   ├── embeddings/
│   └── adversarial/
├── models/                   # Cached/downloaded models
└── README.md
```

---

## ⚙️ Installation

### 🔹 Recommended (Google Colab)

Run directly in Colab with GPU enabled.

### 🔹 Local Setup

```bash
pip install torch torchaudio librosa soundfile matplotlib
pip install transformers datasets
pip install speechbrain epitran phonemizer
```

---

## ▶️ Usage

1. Open `notebook.ipynb`
2. Run all cells sequentially
3. Upload audio when prompted
4. Observe outputs at each stage

---

## 📊 Tasks Covered

| Task     | Description             |
| -------- | ----------------------- |
| Task 1.1 | Language Identification |
| Task 1.2 | ASR with decoding       |
| Task 1.3 | Speech Denoising        |
| Task 2.1 | IPA Mapping             |
| Task 2.2 | Translation             |
| Task 3.1 | Speaker Embedding       |
| Task 4.2 | FGSM Adversarial Attack |

---

## ⚠️ Important Notes

* GPU highly recommended
* First run downloads large models
* Memory-intensive operations (especially embeddings + ASR)
* Must run sequentially

---

## 🎯 Applications

* Multilingual AI assistants
* Code-switched ASR systems
* Speech analytics pipelines
* Robust AI & adversarial research

---

## 🔮 Future Work

* Real-time streaming pipeline
* Improved code-switch detection
* Model optimization for low-resource devices
* Web-based deployment

---

## 📜 License

MIT License

---

## 👨‍💻 Author

**S Kashyap**

---

## 🙏 Acknowledgements

* OpenAI Whisper
* HuggingFace Transformers
* SpeechBrain
* Helsinki-NLP

---

<p align="center">
  ⭐ Star this repository if you found it useful!
</p>
```

