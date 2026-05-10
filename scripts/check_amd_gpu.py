import os
import subprocess
import sys

print("Python:", sys.version)
print("HIP_VISIBLE_DEVICES:", os.getenv("HIP_VISIBLE_DEVICES"))
print("ROCR_VISIBLE_DEVICES:", os.getenv("ROCR_VISIBLE_DEVICES"))

for cmd in [["amd-smi", "list"], ["rocminfo"]]:
    try:
        print(f"\n$ {' '.join(cmd)}")
        out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=20)
        print(out.stdout[:4000])
    except Exception as e:
        print(f"Could not run {' '.join(cmd)}: {e}")

try:
    import torch
    print("\nTorch:", torch.__version__)
    print("Torch HIP:", getattr(torch.version, "hip", None))
    print("CUDA/HIP available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}:", torch.cuda.get_device_name(i))
            x = torch.randn(256, 256, device=f"cuda:{i}", dtype=torch.float16)
            y = x @ x
            print(f"Device {i} matmul OK:", float(y[0, 0].detach().cpu()))
except Exception as e:
    print("Torch check failed:", repr(e))
    raise SystemExit(1)
