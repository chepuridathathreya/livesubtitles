"""Real-time lag test: speech is fed at real speed through the real CaptionEngine.
Reports, for every final caption, how long after the speaker stopped it appeared."""
import argparse, queue, subprocess, sys, time, wave
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
(ROOT / "tmp").mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))
from livecaptions.config import AudioConfig, SegmenterConfig, TranscriberConfig
from livecaptions.engine import CaptionEngine
from livecaptions.transcriber import WhisperTranscriber

ap = argparse.ArgumentParser()
ap.add_argument("--no-partials", action="store_true")
ap.add_argument("--model", default=str(ROOT / "models/telugu-medium"))
ap.add_argument("--lang", default="te")
ap.add_argument("--voice", default="te")
ap.add_argument("--text", default="నాకు తెలుగు బాగా మాట్లాడటం వచ్చు మరియు నేను రేపు కాలేజీకి వెళ్తాను")
ap.add_argument("--max-partial-backlog", type=int, default=None)
ap.add_argument("--beam", type=int, default=3)
args = ap.parse_args()

out = ROOT / "tmp/rt_clip.wav"
subprocess.run(["espeak-ng", "-v", args.voice, "-s", "150", args.text, "-w", str(out)], check=True)
with wave.open(str(out)) as h:
    rate = h.getframerate(); raw = np.frombuffer(h.readframes(h.getnframes()), dtype=np.int16)
a = raw.astype(np.float32) / 32768
a = np.interp(np.arange(0, len(a), rate / 16000), np.arange(len(a)), a).astype(np.float32)
sentence = a
silence = np.zeros(int(16000 * 1.5), np.float32)
audio = np.concatenate([np.zeros(16000, np.float32), sentence, silence, sentence, silence, sentence, silence])
speech_ends = []  # real time (s after start) when each sentence's audio finished
cfg = AudioConfig(); n = cfg.block_samples

class FakeMic:
    def __init__(self): self.q = queue.Queue(); self.t0 = None
    def __enter__(self):
        import threading
        self.t0 = time.perf_counter(); threading.Thread(target=self._feed, daemon=True).start(); return self
    def __exit__(self, *e): pass
    def _feed(self):
        pos = 0
        while pos + n <= len(audio):
            target = self.t0 + (pos + n) / 16000
            time.sleep(max(0, target - time.perf_counter()))
            self.q.put(audio[pos:pos + n]); pos += n
        self.q.put(None)
    def read(self, timeout=0.5):
        try: b = self.q.get(timeout=timeout)
        except queue.Empty: return None
        if b is None: raise SystemExit
        return b
    @property
    def backlog_ms(self): return self.q.qsize() * cfg.block_ms

off = 1.0
for _ in range(3):
    speech_ends.append(off + len(sentence) / 16000); off += len(sentence) / 16000 + 1.5

cpu = {}
tr = WhisperTranscriber(TranscriberConfig(model=args.model, language=args.lang, allowed_languages=(), final_beam=args.beam, **cpu)); tr.warm_up()
mic = FakeMic()
kwargs = {}
engine = CaptionEngine({args.lang: tr}, cfg, SegmenterConfig(), partials=not args.no_partials, language=args.lang, mic_factory=lambda: mic)
if args.max_partial_backlog is not None: engine.max_partial_backlog_ms = args.max_partial_backlog
box = engine.subscribe(); engine.start()
finals = 0; deadline = time.perf_counter() + len(audio) / 16000 + 8; partials = 0; lags = []
while time.perf_counter() < deadline:
    try: ev = box.get(timeout=0.5)
    except queue.Empty: continue
    if ev["type"] != "caption": continue
    now = time.perf_counter() - mic.t0
    if ev["final"]:
        lag = now - ev["voice_end_s"]; finals += 1; lags.append(lag)
        print(f"final {finals}: voice ended {ev['voice_end_s']:5.1f}s | appeared {now:5.1f}s | LAG {lag:4.1f}s | shown delay {ev['latency_ms']} ms | {ev['text'][:36]}", flush=True)
    else: partials += 1
print(f"LAG median {np.median(lags):.1f}s max {max(lags):.1f}s over {len(lags)} captions") if lags else None
print("partials shown:", partials, "| partials enabled:", not args.no_partials)
