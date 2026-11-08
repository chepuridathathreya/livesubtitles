import queue
import threading
import time

from .audio_input import MicrophoneStream
from .config import AudioConfig, SegmenterConfig
from .segmenter import SegmentKind, UtteranceSegmenter
from .transcriber import WhisperTranscriber

LANGUAGES = {"te": "తెలుగు", "hi": "हिन्दी", "en": "English"}


class CaptionEngine:
    """Mic -> segmenter -> transcriber on a background thread, publishing caption events."""

    def __init__(
        self,
        transcribers: dict[str, WhisperTranscriber],
        audio_config: AudioConfig,
        segmenter_config: SegmenterConfig,
        partials: bool = True,
        language: str = "te",
        mic_factory=None,
    ):
        self._transcribers = transcribers
        self._audio_config = audio_config
        self._segmenter_config = segmenter_config
        self._partials = partials
        self._language = language
        self.max_partial_backlog_ms = 300
        self._mic_factory = mic_factory or (lambda: MicrophoneStream(audio_config))
        self._paused = False
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._listeners: set[queue.Queue] = set()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @property
    def state(self) -> dict:
        return {"type": "state", "language": self._language, "paused": self._paused}

    def subscribe(self) -> queue.Queue:
        listener: queue.Queue = queue.Queue(maxsize=200)
        with self._lock:
            self._listeners.add(listener)
        listener.put(self.state)
        return listener

    def unsubscribe(self, listener: queue.Queue) -> None:
        with self._lock:
            self._listeners.discard(listener)

    def _publish(self, event: dict) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener.put_nowait(event)
            except queue.Full:
                pass  # a stalled phone must never slow the microphone loop

    def set_language(self, code: str) -> None:
        if code in self._transcribers:
            self._language = code
            self._publish(self.state)

    def set_paused(self, paused: bool) -> None:
        self._paused = paused
        self._publish(self.state)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _caption(self, segment, final: bool, backlog_ms: int) -> None:
        language = self._language
        result = self._transcribers[language].transcribe(segment.audio, final=final)
        if not result.text:
            return
        # Delay the reader sees: silence waited to confirm the end + audio already queued
        # when decoding began + decoding time.
        latency = result.inference_ms + backlog_ms + (segment.silence_ms if final else 0)
        self._publish(
            {
                "type": "caption",
                "id": int(segment.started_at_s * 1000),
                "final": final,
                "text": result.text,
                "language": language,
                "latency_ms": latency,
                "audio_ms": result.audio_ms,
                "voice_end_s": segment.ended_at_s - segment.silence_ms / 1000,
                "at": time.strftime("%H:%M:%S"),
            }
        )

    def _run(self) -> None:
        segmenter = UtteranceSegmenter(self._audio_config, self._segmenter_config)
        with self._mic_factory() as mic:
            self._publish({"type": "status", "message": "listening"})
            while not self._stop.is_set():
                block = mic.read()
                if block is None:
                    continue
                if self._paused:
                    segmenter.flush()
                    continue
                for segment in segmenter.feed(block):
                    final = segment.kind is SegmentKind.FINAL
                    backlog_ms = mic.backlog_ms
                    if not final and (not self._partials or backlog_ms > self.max_partial_backlog_ms):
                        continue  # a stale draft is worse than none
                    self._caption(segment, final, backlog_ms)
