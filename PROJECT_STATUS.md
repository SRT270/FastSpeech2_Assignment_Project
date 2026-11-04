# Project Status: FastSpeech 2 English and Telugu TTS

- [x] GitHub-ready directory structure matching assignment specifications
- [x] English and Telugu configuration files (`English/config.yaml`, `Telugu/config.yaml`)
- [x] Full dataset manifests generated for Svarah (~5.001 h train, ~1.001 h val) and IndicTTS (~5.001 h train, ~1.001 h val) with zero overlap
- [x] Complete Telugu phonetic G2P, Unicode NFC normalizer, and 68-symbol inventory
- [x] English text normalizer, heteronym disambiguation (E2a vs E2b), and ARPAbet G2P
- [x] Pure PyTorch FastSpeech 2 model (Transformer Encoder, Variance Adaptor, Transformer Decoder, Mel Linear, 5-layer PostNet)
- [x] High-performance, deadlock-free PyTorch audio processing and vocoder synthesis
- [x] Complete preprocessing pipelines (`English/preprocess.py`, `Telugu/preprocess.py`)
- [x] Complete training execution with AdamW and Noam learning rate scheduler (`English/train.py`, `Telugu/train.py`)
- [x] Trained model checkpoints saved (`English/checkpoints/best_model.pth`, `Telugu/checkpoints/best_model.pth`, `checkpoint_20000.pth`)
- [x] Training and validation loss curves generated (`Results/English/`, `Results/Telugu/`)
- [x] Full inference evaluation on all challenge test sentences (E1, E2a, E2b, E3, E4; T1, T2a, T2b, T3, T4)
- [x] High-fidelity 22.05 kHz PCM WAV audio generated for all 10 test cases in `Results/{Language}/test_generated_audios/`
- [x] All 30 variance adaptor plots (duration, pitch, energy) generated in `Results/{Language}/variance_adaptor_plots/`
- [x] Comprehensive, publication-grade Technical Report compiled into `Report/Technical_Report.pdf` and `Report/Technical_Report.md`
- [x] Automated test suite passing (`pytest tests/`)
