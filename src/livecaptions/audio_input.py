import queue
from dataclasses import dataclass

import numpy as np

from .native_libs import register_local_portaudio

register_local_portaudio()

import sounddevice as sd  # noqa: E402

from .config import AudioConfig


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    host_api: str
    channels: int
    default_rate: float


def list_input_devices() -> list[InputDevice]:
    host_apis = sd.query_hostapis()
    devices = []
    for index, info in enumerate(sd.query_devices()):
        if info["max_input_channels"] > 0:
            devices.append(
                InputDevice(
                    index=index,
                    name=info["name"],
                    host_api=host_apis[info["hostapi"]]["name"],
                    channels=info["max_input_channels"],
                    default_rate=info["default_samplerate"],
                )
            )
    return devices


class MicrophoneStream:
    def __init__(self, config: AudioConfig, max_blocks: int = 600):
        self._config = config
        self._blocks: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_blocks)
        self.dropped_blocks = 0
        self._stream = sd.InputStream(
            samplerate=config.sample_rate,
            blocksize=config.block_samples,
            channels=1,
            dtype="float32",
            device=config.device,
            callback=self._on_audio,
        )

    def _on_audio(self, indata, frames, time_info, status) -> None:
        block = indata[:, 0].copy()
        try:
            self._blocks.put_nowait(block)
        except queue.Full:
            self.dropped_blocks += 1

    def __enter__(self):
        self._stream.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._stream.stop()
        self._stream.close()

    def read(self, timeout: float = 0.5) -> np.ndarray | None:
        try:
            return self._blocks.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def backlog_ms(self) -> int:
        return self._blocks.qsize() * self._config.block_ms
