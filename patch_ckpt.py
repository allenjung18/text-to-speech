import torch
import pathlib
torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])

ckpt_path = "train/smoke/lessac-medium.ckpt"
ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")

# Completely remove hyper_parameters to prevent LightningCLI from parsing old args
if "hyper_parameters" in ckpt:
    print("Clearing hyper_parameters dict")
    ckpt["hyper_parameters"] = {}

torch.save(ckpt, "train/smoke/lessac-medium-patched.ckpt")
print("Saved patched checkpoint")
