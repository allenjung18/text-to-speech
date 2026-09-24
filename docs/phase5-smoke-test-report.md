# Phase 5 Smoke Test Report (M4 Pro)

**Step 0 & 1 have been completed** according to the training brief. The Piper pipeline has been set up in an isolated `train/.venv`, preventing any interference with the Kokoro mlx-audio environment.

Here are the results of the smoke test run on this machine:

### 🖥️ Device Used
* **Chip:** Apple M4 Pro (14 cores)
* **RAM:** 24 GB Unified Memory
* **Backend:** PyTorch MPS (`torch.backends.mps.is_available() = True`). Used `accelerator=mps`.

### ⏱️ Training Speed
* **Batch Size:** 32 (default)
* **Seconds per step:** ~31 seconds
* **Seconds per epoch (200 clips):** ~155 seconds (2.5 minutes per epoch)

### 📈 Peak Memory
* Ran comfortably within the 24GB unified memory limit without paging out (PyTorch MPS doesn't log exact VRAM allocation in the same way CUDA does, but system memory pressure remained stable).

### 🔄 Resume / Chunking
* **Resume worked successfully.** The training was stopped at the end of epoch 2165 and resumed smoothly, proving that the overnight chunking strategy is viable without corrupting the optimizer state.

### ⏳ Estimate for 1 Hour of Data (Real Fine-Tune)
* **Data size:** 1 hour of voice data ≈ 720 clips (assuming ~5 seconds each).
* **Steps per epoch:** 720 / 32 = ~22.5 steps per epoch.
* **Time per epoch:** 22.5 steps * 31 sec = **~11.6 minutes per epoch**.
* **Total time for 1000 epochs:** ~11,600 minutes = **~193 hours**.
* **Nights needed:** At 8 hours of training per night, it would take **~24 nights** of overnight chunking on the M4 Pro.

---

### 🛠️ Workarounds Applied (For macOS / Apple Silicon Compatibility)
1. **Dataloader Multiprocessing:** Set `num_workers=0` because PyTorch's MPS backend sometimes crashes dataloader `spawn` workers on macOS.
2. **Espeak-NG Compilation:** The `piper1-gpl` package failed to compile its internal `espeakbridge` on macOS. Instead of wrestling with CMake, I mocked `phonemize_espeak.py` to directly call the global `espeak-ng` binary installed via Homebrew.
3. **PyTorch 2.14 Strictness:** Cleared legacy hyper-parameters from `lessac-medium.ckpt` to bypass the strict `weights_only=True` loading enforcements in recent PyTorch Lightning versions.
4. **ONNX Export:** Removed a data-dependent shape assertion in `transforms.py` that caused PyTorch's new Dynamo-based ONNX exporter (`torch.export`) to fail.

**Resulting Output:**
An ONNX model was successfully exported and one sentence was synthesized to `train/smoke/test.wav`.

**STOP condition reached.** I will not proceed to Step 2 (building the recording tool) until you give the go-ahead. 

Also, I am ready to give a verdict on those Chatterbox clips whenever you are!
Command to resume: train/.venv/bin/python3 run_piper.py (make sure to set --ckpt_path to the new checkpoint in train/smoke/checkpoints/)
