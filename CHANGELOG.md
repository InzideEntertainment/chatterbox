# Changelog

## 2024-12-11
- Force all STFT/ISTFT and Kaldi FBANK paths to run on CPU to avoid cuFFT errors in LXC GPU passthrough; keep model weights on GPU where available.
- Add cache cleanup after each inference (mel/resampler caches, T3 KV cache reset, CUDA allocator flush) to reduce memory growth during multi-run sessions.
- Add `install.sh` bootstrap script for persistent venv setup and CUDA 12.4 torch/torchaudio install inside LXC; includes CUDA sanity check.
