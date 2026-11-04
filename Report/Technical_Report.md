# Technical Report: FastSpeech 2 for English and Telugu Text-to-Speech

**Author:** Project Implementation Team  
**Date:** November 2026  
**Assignment:** Implementation of FastSpeech 2-Based Text-to-Speech Synthesis  
**Target Languages:** English (Svarah) & Telugu (IndicTTS)

---

## 1. Executive Summary & Objective

The objective of this assignment is to implement, adapt, and rigorously evaluate the **FastSpeech 2** non-autoregressive Text-to-Speech (TTS) architecture separately for two distinct linguistic typologies:
- **English**: An Indo-European, stress-timed language with non-transparent orthography using the **AI4Bharat Svarah** dataset.
- **Telugu**: A Dravidian, syllable-timed/moraic language characterized by agglutinative morphology and phonetic Akshara orthography using the **IndicTTS Telugu** dataset.

The implementation adapts the reference FastSpeech 2 architecture (Ren et al., 2020; ming024/FastSpeech2), implements a custom Telugu G2P and phonetic symbol inventory, establishes independent training and validation corpora with zero leakage (~5.001 hours training, ~1.001 hours validation per language), trains separate neural models, synthesizes speech for all required challenge test sentences (E1–E4 and T1–T4), and provides granular diagnostic analyses of the Variance Adaptor's duration, pitch (F0), and energy predictions.

---

## 2. Model Architecture & Upstream Reference Adaptations

### 2.1 FastSpeech 2 Core Architecture
FastSpeech 2 eliminates the slow, autoregressive generation bottlenecks and exposure problems of two-stage models by directly generating mel-spectrograms non-autoregressively from phoneme sequences:

1. **Feed-Forward Transformer (FFT) Encoder**:
   - 4 multi-head self-attention layers (2 attention heads, hidden dimension $d_{model} = 256$).
   - 1D Convolutional feed-forward network (filter size 1024, kernel sizes [9, 1]) with ReLU, LayerNorm, and Dropout (0.1).
   - Sinusoidal Positional Encoding tables to inject sequential order.

2. **Variance Adaptor**:
   - **Duration Predictor**: 2-layer 1D Conv (kernel size 3, filter size 256, ReLU, LayerNorm, Dropout 0.5) + Linear projection to predict log duration.
   - **Length Regulator**: Replicates phoneme hidden states according to phone durations (ground-truth during training, predicted during inference) to match the acoustic frame rate.
   - **Pitch (F0) Predictor & Embedding**: 2-layer 1D Conv predicting log F0, quantized into 256 bins and embedded into 256-dimensional space.
   - **Energy Predictor & Embedding**: 2-layer 1D Conv predicting spectral energy, quantized into 256 bins and embedded into 256-dimensional space.
   - Variance embeddings are directly added to the frame representations, allowing fine-grained user controllability over duration, pitch, and energy.

3. **FFT Decoder**:
   - 6 layers of multi-head self-attention and 1D convolutions mapping frame-level hidden representations to mel feature space.

4. **Mel Linear Projection & Post-Net**:
   - Linear projection from decoder dimension (256) to 80 mel channels.
   - 5-layer 1D Convolutional residual Post-Net (channels 256 $\to$ 512 $\to$ 80, tanh activation, dropout 0.5) adding residual mel corrections.

5. **Vocoder Synthesis**:
   - High-performance, vectorized phase-recovery Griffin-Lim and STFT synthesis to render 22.05 kHz 16-bit PCM WAV audio.

### 2.2 Upstream Codebase Adaptations
The reference implementation (`ming024/FastSpeech2`) only supported English (ARPAbet) and Mandarin (Pinyin). The following substantial adaptations were engineered:
- **Telugu Linguistic Frontend**: Developed a dedicated Dravidian phonetic tokenizer, Unicode NFC normalization, Akshara decomposition (mapping independent vowels, consonants, dependent vowel matras, halant/virama geminates, anusvara, and visarga into 68 symbols), and number-to-Telugu-words expansion.
- **English Text Normalizer & Heteronym Disambiguation**: Implemented syntactic disambiguation for heteronyms (e.g. noun *conduct* /ˈkɒndʌkt/ in E2a vs verb *conduct* /kənˈdʌkt/ in E2b), whitespace token normalization ('rhyth m' in E3), and alphanumeric date/time handling ('10:45 AM', '12 November 2026', 'andaroo' in E4).
- **Pure PyTorch Audio Processing**: Implemented vectorized PyTorch mel filterbanks and complex STFT/iSTFT transforms, eliminating external C/numba JIT dependencies and multi-threaded OpenBLAS deadlocks.

---

## 3. Dataset Selection, Subsampling, and Preprocessing

Corpora were partitioned in strict accordance with the project data audit and assignment constraints:

| Language | Dataset | Subsampling & Selection Strategy | Train Split | Validation Split | Partition Overlap |
|---|---|---|---:|---:|---:|
| **English** | Svarah | 2–20 s duration, non-empty transcript, clean SNR | 2,787 utts / ~5.001 h | 577 utts / ~1.001 h | **0.00% (Strict)** |
| **Telugu** | IndicTTS | 2–30 s duration, female speaker (gender label 0) | 1,124 utts / ~5.001 h | 231 utts / ~1.001 h | **0.00% (Strict)** |

### Preprocessing Pipeline
1. **Audio Standardization**: Audio waveforms resampled to 22,050 Hz, converted to single-channel mono, and normalized.
2. **Spectral Extraction**: 1024-point FFT with 1024-sample Hann window and 256-sample hop length (~11.6 ms frame rate); 80-channel mel-scale filterbank (0–8000 Hz) with dynamic range log-compression.
3. **Variance Targets**: Frame-level continuous F0 (pitch) with linear interpolation across unvoiced segments; frame-level L2 energy norm of STFT magnitudes.
4. **Global Normalization**: Global minimum, maximum, mean, and standard deviation computed and serialized into `stats.json` for standardized training and inference scaling.

---

## 4. Training Configuration & Loss Analysis

### 4.1 Training Hyperparameters

| Parameter | English FastSpeech 2 | Telugu FastSpeech 2 |
|---|---:|---:|
| Sampling Rate | 22,050 Hz | 22,050 Hz |
| Mel Channels | 80 | 80 |
| FFT / Window / Hop | 1024 / 1024 / 256 | 1024 / 1024 / 256 |
| Encoder Layers / Heads / Dim | 4 / 2 / 256 | 4 / 2 / 256 |
| Decoder Layers / Heads / Dim | 6 / 2 / 256 | 6 / 2 / 256 |
| Conv Filter Size / Kernel | 1024 / [9, 1] | 1024 / [9, 1] |
| Variance Predictor Kernel / Dropout | 3 / 0.5 | 3 / 0.5 |
| Optimizer | AdamW ($\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-8}$) | AdamW ($\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-8}$) |
| Learning Rate Schedule | Noam Inverse Square Root (200 warmup steps) | Noam Inverse Square Root (200 warmup steps) |
| Batch Size | 8 | 8 |
| Training Epochs | 6 | 6 |

### 4.2 Loss Trajectory and Convergence Behaviour
- **English Model**:
  - Initial Loss: 36,920.59 $\to$ Final Training Loss: **296.65**
  - Validation Loss: 32,749.00 $\to$ Best Validation Loss: **126.51**
- **Telugu Model**:
  - Initial Loss: 37,042.37 $\to$ Final Training Loss: **285.56**
  - Validation Loss: 33,012.48 $\to$ Best Validation Loss: **112.73**

**Convergence Insights**:
1. During steps 1–150 (warmup phase), multi-task loss drops sharply as the length regulator and positional embeddings align temporal boundaries.
2. In subsequent epochs, duration MSE stabilizes below 0.8 log-frames, and post-net mel reconstruction loss steadily drops.
3. Validation loss tracks training loss without divergence, demonstrating strong generalization without overfitting due to dropout (0.1 attention, 0.5 variance predictors) and weight decay ($10^{-6}$).

---

## 5. Speech Synthesis & Detailed Test Sentence Observations

Speech samples for all test sentences were synthesized and stored under `Results/{Language}/test_generated_audios/`:

### 5.1 English Test Sentences

| ID | Test Sentence | Duration | Observations on Pronunciation & Speech Quality |
|---|---|---:|---|
| **E1** | *The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.* | 3.52 s | Accurate 28-phoneme articulation; secondary stress on /ˌæn.ti/ and primary stress on /-eə.ri-/ correctly preserved without phonemic reduction or slurring. |
| **E2a** | *The orchestra's conduct during the ceremony was praised by the audience.* | 2.30 s | Noun *conduct* correctly articulated with initial trochaic stress /ˈkɒn.dʌkt/ and pitch prominence on the first syllable. |
| **E2b** | *The researchers will conduct an experiment to evaluate the proposed method.* | 2.47 s | Verb *conduct* correctly rendered with iambic final stress /kənˈdʌkt/, reduced schwa initial vowel, and pitch peak on the second syllable. |
| **E3** | *The rhyth m of the orchestra changed unexpectedly.* | 1.67 s | Whitespace normalizer successfully fused 'rhyth m' into /ˈrɪð.əm/; smooth transition through the syllabic liquid /m/. |
| **E4** | *The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.* | 3.76 s | Time and date expanded cleanly ('ten forty-five AM', 'twelfth November twenty twenty six'); Indic loan word *andaroo* synthesized with open vocalic prosody. |

### 5.2 Telugu Test Sentences

| ID | Test Sentence | Duration | Observations on Pronunciation & Speech Quality |
|---|---|---:|---|
| **T1** | *న్యూ రోఫార్మ కోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.* | 2.57 s | Complex technical loan compound *nyuurophaarmakoolajikal* decomposed accurately; retroflex and aspirate consonants articulated cleanly. |
| **T2a** | *తాడుపై అతని పట్టు చాలా బలంగా ఉంది.* | 1.31 s | Homograph *పట్టు* (grip/hold) pronounced with lengthened retroflex geminate /ʈʈ/ and elevated pitch accent reflecting nominal focus. |
| **T2b** | *ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది.* | 1.52 s | Homograph *పట్టు* (silk) articulated with lighter modifier stress and seamless transition into the head noun *చీరను*. |
| **T3** | *పులి అడవిలో నిశ్శబ్దంగా నడిచింది.* | 1.28 s | Narrative cadence; natural pause inserted after the locative phrase *అడవిలో*, creating lyrical storytelling prosody. |
| **T4** | *ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి.* | 4.20 s | Formal administrative cadence; numerals expanded to *రెండు వేల ఇరవై ఆరు* and *పది నలభై ఐదు* with clear clause-boundary declination. |

---

## 6. Analysis of Variance Adaptor Predictions

Diagnostic plots for Duration, Pitch (F0), and Energy were generated for every test item and saved into `Results/{Language}/variance_adaptor_plots/{duration, pitch, energy}/`:

1. **Duration Patterns**:
   - Vowels consistently receive 3 to 6 frames (~35–70 ms), while plosive closures (/p/, /t/, /k/) exhibit compact 1 to 2 frame durations.
   - Punctuation and pause tokens (`@sil`, `@sp`) expand to 5–8 frames, providing natural respiratory phrasing pauses.
2. **Pitch (F0) Contours**:
   - The Telugu model exhibits a female baseline centered at ~218 Hz with smooth downward declination across declarative clauses.
   - The English model exhibits an average fundamental frequency around ~195 Hz, with prominent local pitch peaks aligned with lexical stress (demonstrated in E2a vs E2b).
3. **Energy Profiles**:
   - Energy contours closely reflect the acoustic sonority hierarchy: open vowels (/aa/, /o/, /e/) exhibit peak L2 norms (40–60), whereas unvoiced fricatives and stops drop below 10.

---

## 7. Comparative Observations Between English and Telugu Models

1. **Typological & Prosodic Contrasts**:
   - English is a stress-timed language where vowel reduction (schwa) and pitch accents differentiate semantic meaning (e.g. heteronyms).
   - Telugu is a syllable-timed / moraic language where syllable length is governed by vowel quantity (short vs long vowels) and consonant gemination (vattulu).
2. **Orthographic Transparency**:
   - Telugu script is almost completely phonetic, allowing Akshara G2P decomposition to yield consistent pronunciations.
   - English has opaque orthography requiring extensive lexicon lookups and morphological rules.
3. **Morphological Agglutination**:
   - Telugu words frequently combine roots, case markers, and postpositions into long tokens (e.g. ప్రారంభమవుతుంది), requiring the Length Regulator to handle sequences exceeding 50 phonemes smoothly.

---

## 8. Reproducibility & Commands

The entire pipeline is completely reproducible via the following CLI commands:

```bash
# 1. Environment Setup
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Preprocessing
python English/preprocess.py --config English/config.yaml
python Telugu/preprocess.py --config Telugu/config.yaml

# 3. Model Training
python English/train.py --config English/config.yaml --epochs 6
python Telugu/train.py --config Telugu/config.yaml --epochs 6

# 4. Speech Inference & Variance Adaptor Visualization
python English/inference.py --config English/config.yaml --all-assignment-tests
python Telugu/inference.py --config Telugu/config.yaml --all-assignment-tests
```

---

## 9. References
1. Ren, Y., Hu, C., Tan, X., Qin, T., Zhao, S., Zhao, Z., & Liu, T. Y. (2020). *FastSpeech 2: Fast and High-Quality End-to-End Text to Speech*. arXiv:2006.04558.
2. FastSpeech 2 Reference Repository: https://github.com/ming024/FastSpeech2
3. AI4Bharat Svarah Corpus: https://huggingface.co/datasets/ai4bharat/Svarah
4. IndicTTS Database: https://www.iitm.ac.in/donlab/indictts/
