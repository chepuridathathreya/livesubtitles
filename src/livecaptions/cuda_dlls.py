import os
import sys
from pathlib import Path


def register_cuda_dlls() -> None:
    if sys.platform != "win32":
        return
    for base in map(Path, sys.path):
        nvidia_root = base / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for bin_dir in nvidia_root.glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]
