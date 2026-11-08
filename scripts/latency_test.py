"""Speed + coverage test. Synthetic espeak clips go through the real segmenter and
transcribers. latency = silence wait (endpoint) + decode time, which is what a reader waits
after the speaker stops. Robotic TTS is a rough proxy, not a human-voice benchmark."""
import argparse, subprocess, sys, wave
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
(ROOT / "tmp").mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))
from livecaptions.config import AudioConfig, SegmenterConfig, TranscriberConfig
from livecaptions.segmenter import SegmentKind, UtteranceSegmenter
from livecaptions.transcriber import WhisperTranscriber

ap = argparse.ArgumentParser()
ap.add_argument("--cpu", action="store_true")
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--beam", type=int, default=3)
ap.add_argument("--general", default="large-v3-turbo")
ap.add_argument("--te", default=str(ROOT / "models/telugu-medium"))
args = ap.parse_args()
cpu = {"device": "cpu", "compute_type": "int8", "cpu_threads": args.threads, "final_beam": args.beam} if args.cpu else {}

CLIPS = [  # (language, espeak voice, text, words that must appear)
    ("en", "en", "Hello, can you hear me clearly", ["hello", "hear", "clearly"]),
    ("en", "en", "yes yes I said yes", ["yes", "yes", "said", "yes"]),
    ("hi", "hi", "नमस्ते आप कैसे हैं", ["नमस्ते", "कैसे", "हैं"]),
    ("te", "te", "నాకు తెలుగు అర్థం అవుతుంది", ["నాకు", "తెలుగు", "అర్థం"]),
    ("te", "te", "రేపు రేపు కాలేజీ ఉంది", ["రేపు", "రేపు", "కాలేజీ"]),
]
tmp = ROOT / "tmp"
def load(voice, text):
    out = tmp / "lat_clip.wav"
    subprocess.run(["espeak-ng", "-v", voice, "-s", "140", text, "-w", str(out)], check=True)
    with wave.open(str(out)) as h:
        rate = h.getframerate(); raw = np.frombuffer(h.readframes(h.getnframes()), dtype=np.int16)
    a = raw.astype(np.float32) / 32768
    a = np.interp(np.arange(0, len(a), rate / 16000), np.arange(len(a)), a).astype(np.float32)
    return np.concatenate([np.zeros(16000, np.float32), a, np.zeros(24000, np.float32)])

print("loading models (cpu=%s)..." % args.cpu, flush=True)
general = WhisperTranscriber(TranscriberConfig(model=args.general, language="en", allowed_languages=(), **cpu))
models = {"en": general, "hi": general.with_language("hi"),
          "te": WhisperTranscriber(TranscriberConfig(model=args.te, language="te", allowed_languages=(), **cpu))}
for m in models.values(): m.warm_up()

audio_cfg, seg_cfg = AudioConfig(), SegmenterConfig()
tot_words = hit_words = 0; lats = []
for lang, voice, text, must in CLIPS:
    audio = load(voice, text); seg = UtteranceSegmenter(audio_cfg, seg_cfg); finals = []
    for i in range(0, len(audio) - audio_cfg.block_samples, audio_cfg.block_samples):
        for s in seg.feed(audio[i:i + audio_cfg.block_samples]):
            if s.kind is SegmentKind.FINAL:
                r = models[lang].transcribe(s.audio, final=True); finals.append(r)
    got = " ".join(r.text for r in finals)
    # coverage: each required word must be matched, repeats must appear that many times
    pool = got.lower().translate({ord(c): " " for c in ",.?!।…;:\"'"}).split(); hits = 0
    for w in must:
        if w.lower() in pool: pool.remove(w.lower()); hits += 1
    tot_words += len(must); hit_words += hits
    for r in finals: lats.append(r.inference_ms + seg_cfg.endpoint_silence_ms)
    print(f"[{lang}] captions={len(finals)} words {hits}/{len(must)} "
          f"latency={[r.inference_ms + seg_cfg.endpoint_silence_ms for r in finals]} ms  -> {got!r}", flush=True)
print(f"COVERAGE {hit_words}/{tot_words} words | LATENCY median {int(np.median(lats))} ms max {max(lats)} ms")
