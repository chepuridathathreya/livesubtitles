from dataclasses import dataclass


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    block_ms: int = 100
    device: int | str | None = None

    @property
    def block_samples(self) -> int:
        return self.sample_rate * self.block_ms // 1000


@dataclass(frozen=True)
class SegmenterConfig:
    preroll_blocks: int = 8
    min_rms: float = 0.004
    noise_factor: float = 2.5
    noise_window_blocks: int = 30
    noise_percentile: float = 10.0
    endpoint_silence_ms: int = 700
    max_utterance_s: float = 8.0
    # (utterance seconds so far, silence needed to end it): long speech may end at short pauses
    soft_endpoints: tuple[tuple[float, int], ...] = ((3.0, 450), (5.0, 250))
    min_utterance_ms: int = 300
    partial_interval_s: float = 1.0


@dataclass(frozen=True)
class TranscriberConfig:
    model: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str | None = None
    allowed_languages: tuple[str, ...] = ("en", "hi", "te")
    cpu_threads: int = 0
    partial_beam: int = 1
    final_beam: int = 3
    max_no_speech_prob: float = 0.6
    min_avg_logprob: float = -1.0
    short_noise_words: tuple[str, ...] = ("you", "thank you", "thanks for watching")
    short_noise_max_ms: int = 1600
    filler_words: tuple[str, ...] = ("ఉమ్", "ఆహ్", "మ్మ్", "మ్", "హ్మ్", "um", "uh", "hmm", "उम्", "हम्म")
    filler_max_ms: int = 2500
    loop_min_repeats: int = 6
    loop_keep: int = 2
