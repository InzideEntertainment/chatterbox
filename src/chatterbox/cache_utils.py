import torch

from chatterbox.models.s3gen.s3gen import get_resampler
from chatterbox.models.s3gen.utils.mel import clear_mel_cache
from chatterbox.models.voice_encoder import melspec as ve_melspec


def clear_inference_caches():
    """Release cached audio helpers and torch GPU cache at the end of a job."""
    clear_mel_cache()
    get_resampler.cache_clear()
    ve_melspec.mel_basis.cache_clear()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
