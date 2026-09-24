import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path
import difflib

import numpy as np
import sounddevice as sd
import soundfile as sf
import mlx_whisper

from audiobook_maker import words_for_diff

def get_prompts(lexicon_path: Path):
    prompts = []
    
    # Lexicon names
    if lexicon_path.exists():
        try:
            lexicon = json.loads(lexicon_path.read_text())
            i = 1
            for name in lexicon:
                if name != "_comment":
                    prompts.append((f"lexicon_{i:04d}", f"{name}."))
                    prompts.append((f"lexicon_{i+1:04d}", f"This is {name}."))
                    i += 2
        except Exception as e:
            print(f"Error loading lexicon: {e}")
            
    # ARCTIC
    voices_dir = Path("voices")
    voices_dir.mkdir(parents=True, exist_ok=True)
    arctic_cache = voices_dir / "cmuarctic.data"
    if not arctic_cache.exists():
        print("Downloading CMU ARCTIC prompts...")
        urllib.request.urlretrieve("http://festvox.org/cmu_arctic/cmuarctic.data", arctic_cache)
    
    for line in arctic_cache.read_text().splitlines():
        match = re.match(r'\( (arctic_[a-z0-9]+) "(.*)" \)', line)
        if match:
            prompts.append((match.group(1), match.group(2)))
            
    return prompts

def get_unrecorded_prompts(prompts, metadata_csv: Path):
    recorded_ids = set()
    if metadata_csv.exists():
        with open(metadata_csv, "r", encoding="utf-8") as f:
            for line in f:
                if "|" in line:
                    recorded_ids.add(line.split("|")[0])
    return [p for p in prompts if p[0] not in recorded_ids]

def append_metadata(metadata_csv: Path, pid: str, text: str):
    with open(metadata_csv, "a", encoding="utf-8") as f:
        f.write(f"{pid}|{text}\n")

def check_session(session_recorded):
    if not session_recorded:
        return
        
    print("\n--- Running Whisper check on session ---")
    print("This may take a moment to load the model...")
    model = "mlx-community/whisper-large-v3-turbo"
    
    for pid, wav_path, text, audio in session_recorded:
        peak = np.max(np.abs(audio))
        peak_db = 20 * np.log10(peak + 1e-9)
        
        # noise check: rms of lowest 10% blocks
        frame_len = 1024
        frames = [np.mean(audio[i:i+frame_len]**2) for i in range(0, len(audio)-frame_len, frame_len)]
        noise_floor = 10 * np.log10(np.percentile(frames, 10) + 1e-9) if frames else -100
            
        is_clipping = peak_db > -1.0
        is_noisy = noise_floor > -45.0 # threshold for noise
        
        res = mlx_whisper.transcribe(
            str(wav_path), 
            path_or_hf_repo=model, 
            word_timestamps=True,
            initial_prompt=text, 
            condition_on_previous_text=False
        )
        
        heard = [(w["word"], w["start"]) for seg in res["segments"] for w in seg.get("words", [])]
        hyp = []
        for raw, start in heard:
            for w in words_for_diff(raw):
                hyp.append(w)
                
        src = words_for_diff(text)
        
        matcher = difflib.SequenceMatcher(None, src, hyp)
        edits = 0
        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op != "equal":
                edits += max(i2 - i1, j2 - j1)
        
        wer = edits / max(1, len(src))
        
        flags = []
        if wer > 0.1: flags.append(f"WER={wer:.1%}")
        if is_clipping: flags.append(f"Clipping (peak={peak_db:.1f}dB)")
        if is_noisy: flags.append(f"Noisy (floor={noise_floor:.1f}dB)")
        
        if flags:
            print(f"[{pid}] FLAG: {', '.join(flags)}")
            print(f"  Text:  {text}")
            print(f"  Heard: {' '.join(hyp)}")
            print(f"  File:  {wav_path}")

def record_cli(dataset_dir: Path, lexicon_path: Path):
    prompts = get_prompts(lexicon_path)
    wavs_dir = dataset_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    metadata_csv = dataset_dir / "metadata.csv"
    
    unrecorded = get_unrecorded_prompts(prompts, metadata_csv)
    fs = 22050
    session_recorded = []
    
    print(f"Found {len(prompts)} prompts. {len(prompts) - len(unrecorded)} already recorded.")
    
    try:
        for pid, text in unrecorded:
            while True:
                cmd = input(f"\n[{pid}] {text}\nPress Enter to start (q to quit)... ")
                if cmd.lower() == 'q':
                    check_session(session_recorded)
                    return
                
                print("Recording... Press Enter to stop.")
                recording = []
                def callback(indata, frames, time_info, status):
                    recording.append(indata.copy())
                stream = sd.InputStream(samplerate=fs, channels=1, callback=callback)
                with stream:
                    input()
                
                if not recording:
                    print("No audio recorded.")
                    continue
                    
                audio = np.concatenate(recording, axis=0)
                
                cmd = input("Keep (Enter), Redo (r), Quit (q)? ")
                if cmd.lower() == 'q':
                    check_session(session_recorded)
                    return
                if cmd.lower() == 'r':
                    continue
                
                # Keep
                wav_path = wavs_dir / f"{pid}.wav"
                sf.write(wav_path, audio, fs)
                append_metadata(metadata_csv, pid, text)
                session_recorded.append((pid, wav_path, text, audio))
                break
    except KeyboardInterrupt:
        pass
        
    check_session(session_recorded)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("voices/dataset"))
    parser.add_argument("--lexicon", type=Path, default=Path(os.path.expanduser("~/Desktop/ShadowSlave/lexicon.json")))
    args = parser.parse_args()
    
    record_cli(args.dataset, args.lexicon)
