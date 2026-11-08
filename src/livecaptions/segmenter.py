from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .config import AudioConfig, SegmenterConfig


class SegmentKind(Enum):
    PARTIAL = "partial"
    FINAL = "final"


@dataclass(frozen=True)
class Segment:
    kind: SegmentKind
    audio: np.ndarray
    started_at_s: float
    ended_at_s: float
    silence_ms: int = 0


class UtteranceSegmenter:
    def __init__(self, audio: AudioConfig, config: SegmenterConfig):
        self._audio = audio
        self._config = config
        self._preroll: deque[np.ndarray] = deque(maxlen=config.preroll_blocks)
        self._buffer: list[np.ndarray] = []
        self._recent_rms: deque[float] = deque(maxlen=config.noise_window_blocks)
        self._noise_floor = 0.0
        self._silence_ms = 0
        self._since_partial_s = 0.0
        self._clock_s = 0.0
        self._started_at_s = 0.0
        self._speaking = False

    @property
    def noise_floor(self) -> float:
        return self._noise_floor

    @property
    def speaking(self) -> bool:
        return self._speaking

    def feed(self, block: np.ndarray) -> list[Segment]:
        block_s = self._audio.block_ms / 1000
        self._clock_s += block_s
        rms = float(np.sqrt(np.mean(np.square(block))))
        self._recent_rms.append(rms)
        if len(self._recent_rms) >= 10:
            self._noise_floor = float(
                np.percentile(self._recent_rms, self._config.noise_percentile)
            )
        threshold = max(self._config.min_rms, self._noise_floor * self._config.noise_factor)
        is_speech = rms > threshold and len(self._recent_rms) >= 10

        if not self._speaking:
            if is_speech:
                self._speaking = True
                self._started_at_s = self._clock_s - block_s * (len(self._preroll) + 1)
                self._buffer = [*self._preroll, block]
                self._silence_ms = 0
                self._since_partial_s = 0.0
            else:
                self._preroll.append(block)
            return []

        self._buffer.append(block)
        self._silence_ms = 0 if is_speech else self._silence_ms + self._audio.block_ms
        self._since_partial_s += block_s
        duration_s = len(self._buffer) * block_s

        needed_ms = self._config.endpoint_silence_ms
        for after_s, silence_ms in self._config.soft_endpoints:
            if duration_s >= after_s:
                needed_ms = min(needed_ms, silence_ms)
        ended = self._silence_ms >= needed_ms or duration_s >= self._config.max_utterance_s
        if ended:
            return self._finish()

        if self._since_partial_s >= self._config.partial_interval_s:
            self._since_partial_s = 0.0
            return [self._snapshot(SegmentKind.PARTIAL)]
        return []

    def flush(self) -> list[Segment]:
        return self._finish() if self._speaking else []

    def _snapshot(self, kind: SegmentKind) -> Segment:
        return Segment(
            kind=kind,
            audio=np.concatenate(self._buffer),
            started_at_s=self._started_at_s,
            ended_at_s=self._clock_s,
            silence_ms=self._silence_ms,
        )

    def _finish(self) -> list[Segment]:
        segment = self._snapshot(SegmentKind.FINAL)
        self._buffer = []
        self._preroll.clear()
        self._speaking = False
        self._silence_ms = 0
        voiced_ms = (segment.ended_at_s - segment.started_at_s) * 1000 - segment.silence_ms
        if voiced_ms < self._config.min_utterance_ms:
            return []
        return [segment]
