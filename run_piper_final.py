import sys
import torch
import pathlib

# Allow PosixPath for weights_only=True loading
torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])

import runpy
sys.argv = [
  "piper.train",
  "fit",
  "--data.voice_name", "gemini",
  "--data.csv_path", "voices/dataset/metadata.csv",
  "--data.audio_dir", "voices/dataset/wavs/",
  "--model.sample_rate", "22050",
  "--data.espeak_voice", "en-us",
  "--data.cache_dir", "train/real/cache",
  "--data.config_path", "train/real/config.json",
  "--data.batch_size", "8",
  "--ckpt_path", "train/real/lessac-medium-patched.ckpt",
  "--trainer.max_epochs", "4000",
  "--trainer.default_root_dir", "train/real/checkpoints",
  "--trainer.accelerator", "mps", 
  "--data.num_workers", "0"
]

runpy.run_module("piper.train", run_name="__main__")
