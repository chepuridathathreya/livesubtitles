import copy
import time
from dataclasses import dataclass, replace

import numpy as np
from faster_whisper import WhisperModel

from .config import TranscriberConfig
from .cuda_dlls import register_cuda_dlls


@dataclass(frozen=True)
class Transcription:
    text: str
    language: str
    language_probability: float
    inference_ms: int
    audio_ms: int


def collapse_loops(text: str, min_repeats: int, keep: int) -> str:
    """Whisper sometimes loops ("ల్లు ల్లు ల్లు ..."). Shrink a pattern of 1-3 words that
    repeats min_repeats or more times in a row down to `keep` copies. A genuine repeat
    like "yes yes" is far below the threshold and is left alone."""
    words = text.split()
    for size in (1, 2, 3):
        out: list[str] = []
        i = 0
        while i < len(words):
            unit = words[i : i + size]
            count = 1
            while len(unit) == size and words[i + count * size : i + (count + 1) * size] == unit:
                count += 1
            if count >= min_repeats:
                out.extend(unit * keep)
                i += count * size
            else:
                out.append(words[i])
                i += 1
        words = out
    return " ".join(words)


class WhisperTranscriber:
    def __init__(self, config: TranscriberConfig):
        if config.device == "cuda":
            register_cuda_dlls()
        self._config = config
        self._model = WhisperModel(
            config.model,
            device=config.device,
            compute_type=config.compute_type,
            cpu_threads=config.cpu_threads,
        )

    def with_language(self, language: str | None) -> "WhisperTranscriber":
        """A second transcriber for another language that reuses the loaded model."""
        clone = copy.copy(self)
        clone._config = replace(self._config, language=language, allowed_languages=())
        return clone

    def warm_up(self) -> None:
        self.transcribe(np.zeros(16000, dtype=np.float32), final=False)

    def _decode(self, audio: np.ndarray, final: bool, language: str | None):
        return self._model.transcribe(
            audio,
            language=language,
            beam_size=self._config.final_beam if final else self._config.partial_beam,
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True,
            temperature=0.0,
        )

    def transcribe(self, audio: np.ndarray, final: bool) -> Transcription:
        started = time.perf_counter()
        segments, info = self._decode(audio, final, self._config.language)
        allowed = self._config.allowed_languages
        if self._config.language is None and allowed and info.language not in allowed:
            # Auto-detect picked a language we don't support: take the best allowed one
            # and decode again (segments are lazy, so the first decode cost only detection).
            probs = dict(info.all_language_probs or [])
            best = max(allowed, key=lambda code: probs.get(code, 0.0))
            total = sum(probs.get(code, 0.0) for code in allowed) or 1.0
            segments, info = self._decode(audio, final, best)
            language, probability = best, probs.get(best, 0.0) / total
        else:
            language, probability = info.language, info.language_probability
        kept = [
            segment.text.strip()
            for segment in segments
            if not (
                segment.no_speech_prob > self._config.max_no_speech_prob
                and segment.avg_logprob < self._config.min_avg_logprob
            )
        ]
        text = collapse_loops(
            " ".join(part for part in kept if part),
            self._config.loop_min_repeats,
            self._config.loop_keep,
        )
        audio_ms = int(len(audio) / 16000 * 1000)
        if (
            audio_ms <= self._config.short_noise_max_ms
            and text.strip(" .!?,").lower() in self._config.short_noise_words
        ):
            text = ""  # Whisper's usual hallucination on a click or breath
        if audio_ms <= self._config.filler_max_ms and text.strip(" .!?,…") in self._config.filler_words:
            text = ""  # a hum or "um" is not a caption
        return Transcription(
            text=text,
            language=language,
            language_probability=probability,
            inference_ms=int((time.perf_counter() - started) * 1000),
            audio_ms=audio_ms,
        )
