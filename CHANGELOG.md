# Changelog

## 2025-12-12
- Set Gradio apps to local-only (no public share links) and bind to 0.0.0.0 with explicit ports.
- Document fresh reinstall steps for LXC hosts (wipe old repo/venv, clone, run install.sh, activate venv, launch apps).
- Limit TTS Gradio textbox to 150 chars to constrain input length.
- Add `conversation.py` JSON runner with path resolution and `conversation_local.py` inline template.

## 2025-12-11
- Force all STFT/ISTFT and Kaldi FBANK paths to run on CPU to avoid cuFFT errors in LXC GPU passthrough; keep model weights on GPU where available.
- Add cache cleanup after each inference (mel/resampler caches, T3 KV cache reset, CUDA allocator flush) to reduce memory growth during multi-run sessions.
- Add `install.sh` bootstrap script for persistent venv setup and CUDA 12.4 torch/torchaudio install inside LXC; includes CUDA sanity check.
