import os
import csv
import soundfile as sf
import mlx_whisper
from pathlib import Path

wav_path = "/tmp/gemini_long4.wav"
audio, fs = sf.read(wav_path)

dataset_dir = Path("voices/dataset")
wavs_dir = dataset_dir / "wavs"
wavs_dir.mkdir(parents=True, exist_ok=True)
metadata_path = dataset_dir / "metadata.csv"

res = mlx_whisper.transcribe(
    wav_path,
    path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
    word_timestamps=True,
    condition_on_previous_text=True
)

metadata = []
base_idx = 0
if metadata_path.exists():
    with open(metadata_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("gemini_"):
                try:
                    num = int(line.split("_")[1].split("|")[0])
                    if num > base_idx:
                        base_idx = num
                except:
                    pass

for i, seg in enumerate(res["segments"]):
    base_idx += 1
    pid = f"gemini_{base_idx:04d}"
    text = seg["text"].strip()
    
    if not text:
        continue
        
    start_time = max(0, seg["start"] - 0.1) # small pad
    end_time = min(len(audio)/fs, seg["end"] + 0.1)
    
    start_sample = int(start_time * fs)
    end_sample = int(end_time * fs)
    
    chunk = audio[start_sample:end_sample]
    out_wav = wavs_dir / f"{pid}.wav"
    sf.write(out_wav, chunk, fs)
    
    metadata.append(f"{pid}|{text}")
    if i % 50 == 0:
        print(f"[{pid}] {text}")

with open(metadata_path, "a", encoding="utf-8") as f:
    f.write("\n".join(metadata) + "\n")

print(f"Created {len(metadata)} new Gemini slices! Appended to metadata.")
