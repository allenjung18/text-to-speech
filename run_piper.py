import sys
import torch
import pathlib

# Allow PosixPath for weights_only=True loading
torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])

import runpy
sys.argv = [
  "piper.train",
  "fit",
  "--data.voice_name", "smoke",
  "--data.csv_path", "train/smoke/LJSpeech/metadata.csv",
  "--data.audio_dir", "train/smoke/LJSpeech/wavs/",
  "--model.sample_rate", "22050",
  "--data.espeak_voice", "en-us",
  "--data.cache_dir", "train/smoke/cache",
  "--data.config_path", "train/smoke/config.json",
  "--data.batch_size", "32",
  "--ckpt_path", "train/smoke/lessac-medium-patched.ckpt",
  "--trainer.max_epochs", "2166",
  "--trainer.default_root_dir", "train/smoke/checkpoints",
  "--trainer.accelerator", "mps", "--data.num_workers", "0"
]

runpy.run_module("piper.train", run_name="__main__")
