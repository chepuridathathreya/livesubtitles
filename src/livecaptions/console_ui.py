import shutil
import sys

from .transcriber import Transcription


def _clock(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class ConsoleCaptions:
    def __init__(self):
        self._partial_visible = False

    def _clear_partial(self) -> None:
        if self._partial_visible:
            sys.stdout.write("\r" + " " * (shutil.get_terminal_size().columns - 1) + "\r")
            self._partial_visible = False

    def status(self, message: str) -> None:
        self._clear_partial()
        sys.stdout.write(f"{message.upper()}\n")
        sys.stdout.flush()

    def partial(self, started_at_s: float, result: Transcription) -> None:
        if not result.text:
            return
        width = shutil.get_terminal_size().columns - 1
        line = f"[{_clock(started_at_s)}] ... {result.text}"
        sys.stdout.write("\r" + line[-width:].ljust(width))
        sys.stdout.flush()
        self._partial_visible = True

    def final(self, started_at_s: float, result: Transcription) -> None:
        self._clear_partial()
        if not result.text:
            return
        meta = (
            f"{result.language.upper()} {result.language_probability * 100:.0f}%"
            f" / {result.inference_ms} MS FOR {result.audio_ms} MS"
        )
        sys.stdout.write(f"[{_clock(started_at_s)}] {result.text}\n")
        sys.stdout.write(f"           {meta}\n")
        sys.stdout.flush()
