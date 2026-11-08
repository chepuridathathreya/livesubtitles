"""Linux only: make a locally extracted PortAudio (vendor/portaudio) visible to sounddevice.

sounddevice looks the library up with ctypes.util.find_library, which ignores
LD_LIBRARY_PATH. When libportaudio2 is not installed system-wide (no sudo), we point
find_library at the copy under vendor/portaudio instead.
"""
import ctypes
import ctypes.util
import sys
from pathlib import Path

_VENDOR = (
    Path(__file__).resolve().parents[2]
    / "vendor/portaudio/usr/lib/x86_64-linux-gnu/libportaudio.so.2"
)


def register_local_portaudio() -> None:
    if not sys.platform.startswith("linux") or not _VENDOR.exists():
        return
    if ctypes.util.find_library("portaudio"):
        return
    original = ctypes.util.find_library

    def patched(name: str):
        return str(_VENDOR) if name == "portaudio" else original(name)

    ctypes.util.find_library = patched
