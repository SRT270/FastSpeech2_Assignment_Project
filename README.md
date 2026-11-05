# FastSpeech 2: English and Telugu Text-to-Speech Synthesis

Complete implementation, training, and evaluation of **FastSpeech 2** for **English** (AI4Bharat Svarah) and **Telugu** (IndicTTS Telugu) Text-to-Speech synthesis.

**GitHub repository:** https://github.com/SRT270/FastSpeech2-Assignment

---

## Deliverables & Directory Structure

```text
project/
├── English/
│   ├── train.py                  # Training pipeline with AdamW & Noam schedule
│   ├── preprocess.py             # Feature extraction and manifest formatting
│   ├── inference.py              # Test synthesis & variance adaptor plotting
│   ├── config.yaml               # English model & data hyperparameters
│   └── checkpoints/
│       ├── best_model.pth        # Best validation model checkpoint
│       └── checkpoint_20000.pth  # Trained model checkpoint
├── Telugu/
│   ├── train.py                  # Training pipeline with Telugu phonetics
│   ├── preprocess.py             # Telugu Akshara G2P & feature extraction
│   ├── inference.py              # Test synthesis & variance adaptor plotting
│   ├── config.yaml               # Telugu model & data hyperparameters
│   └── checkpoints/
│       ├── best_model.pth        # Best validation model checkpoint
│       └── checkpoint_20000.pth  # Trained model checkpoint
├── Results/
│   ├── English/
│   │   ├── training_loss.png     # Training loss convergence curve
│   │   ├── validation_loss.png   # Validation loss trajectory
│   │   ├── test_generated_audios/
│   │   │   ├── E1.wav, E2a.wav, E2b.wav, E3.wav, E4.wav
│   │   └── variance_adaptor_plots/
│   │       ├── duration/         # E1.png - E4.png
│   │       ├── pitch/            # E1.png - E4.png
│   │       └── energy/           # E1.png - E4.png
│   └── Telugu/
│       ├── training_loss.png     # Training loss convergence curve
│       ├── validation_loss.png   # Validation loss trajectory
│       ├── test_generated_audios/
│       │   ├── T1.wav, T2a.wav, T2b.wav, T3.wav, T4.wav
│       │   └── variance_adaptor_plots/
│       │       ├── duration/     # T1.png - T4.png
│       │       ├── pitch/        # T1.png - T4.png
│       │       └── energy/       # T1.png - T4.png
├── Report/
│   ├── Technical_Report.pdf      # Complete publication-grade technical report
│   └── Technical_Report.md       # Markdown report source
├── fastspeech2/                  # Core modular PyTorch package
│   ├── model.py                  # Transformer Encoder, Decoder, PostNet, MelLinear
│   ├── variance_adaptor.py       # DurationPredictor, LengthRegulator, Pitch/Energy
│   ├── audio.py                  # Fast PyTorch STFT, Mel-Filterbank, Vocoder
│   ├── loss.py                   # Multi-task loss (Mel Pre/Post, Dur, Pitch, Energy)
│   ├── dataset.py                # Dataset and batch collation
│   └── text/                     # English and Telugu G2P and phonetic symbol tables
├── data/
│   ├── manifests/                # Svarah and IndicTTS train/val CSV manifests
│   └── test_sentences.json       # Challenge test sentences E1-E4 and T1-T4
├── README.md
└── requirements.txt
```

---

## Datasets & Split Statistics

- **English (Svarah)**:
  - Training: 2,787 utterances (~5.001 hours), duration filter 2–20s.
  - Validation: 577 utterances (~1.001 hours), duration filter 2–20s.
  - Leakage: **Strictly 0.00% overlap**.
- **Telugu (IndicTTS)**:
  - Training: 1,124 utterances (~5.001 hours), female speaker (gender 0), duration filter 2–30s.
  - Validation: 231 utterances (~1.001 hours), duration filter 2–30s.
  - Leakage: **Strictly 0.00% overlap**.

---

## Quickstart & Reproducibility

### 1. Environment Installation
```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Preprocessing
Run preprocessing to generate phone sequences, spectrogram features, and normalization statistics (`stats.json`):
```bash
python English/preprocess.py --config English/config.yaml
python Telugu/preprocess.py --config Telugu/config.yaml
```

### 3. Model Training
Train the English and Telugu models independently:
```bash
python English/train.py --config English/config.yaml --epochs 6
python Telugu/train.py --config Telugu/config.yaml --epochs 6
```
Training generates checkpoints in `{Language}/checkpoints/` and loss plots in `Results/{Language}/`.

### 4. Speech Inference & Variance Adaptor Visualization
Generate 22.05 kHz PCM WAV speech and duration/pitch/energy plots for all test sentences:
```bash
python English/inference.py --config English/config.yaml --all-assignment-tests
python Telugu/inference.py --config Telugu/config.yaml --all-assignment-tests
```

### 5. Run Verification Tests
```bash
pytest tests/
```

---

## Challenge Test Sentences

### English Test Sentences
- **E1**: *The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.*
- **E2a**: *The orchestra's conduct during the ceremony was praised by the audience.* (Noun stress: /ˈkɒndʌkt/)
- **E2b**: *The researchers will conduct an experiment to evaluate the proposed method.* (Verb stress: /kənˈdʌkt/)
- **E3**: *The rhyth m of the orchestra changed unexpectedly.* (Whitespace-split token normalization)
- **E4**: *The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.* (Alphanumeric date/time and Indic loan word)

### Telugu Test Sentences
- **T1**: *న్యూ రోఫార్మ కోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.* (Technical loan compound *nyuurophaarmakoolajikal*)
- **T2a**: *తాడుపై అతని పట్టు చాలా బలంగా ఉంది.* (Homograph *పట్టు* as noun/grip)
- **T2b**: *ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది.* (Homograph *పట్టు* as modifier/silk)
- **T3**: *పులి అడవిలో నిశ్శబ్దంగా నడిచింది.* (Lyrical narrative prosody)
- **T4**: *ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి.* (Formal administrative speech with number expansions)

---

## Technical Report

A detailed, publication-grade report is available in [Report/Technical_Report.pdf](file:///Report/Technical_Report.pdf) (source in [Report/Technical_Report.md](file:///Report/Technical_Report.md)). It covers:
- Complete architectural analysis and modifications from upstream.
- Subsampling strategy and preprocessing pipeline.
- Training loss convergence and stability analysis.
- Linguistic and acoustic evaluations of all 10 synthesized test samples.
- Variance Adaptor duration, pitch (F0), and energy diagnostic plots.
- Cross-lingual comparisons between stress-timed English and syllable-timed Telugu.
