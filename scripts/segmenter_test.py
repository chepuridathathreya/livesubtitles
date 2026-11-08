import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from livecaptions.config import AudioConfig, SegmenterConfig
from livecaptions.segmenter import SegmentKind, UtteranceSegmenter

rng = np.random.default_rng(0)
def run(noise_rms, speech_rms):
    a = AudioConfig(); seg = UtteranceSegmenter(a, SegmenterConfig()); n = a.block_samples
    finals = []
    # 3 s noise, 2 s speech, 2 s noise, 3 s speech, 3 s noise
    plan = [(30, 0), (20, 1), (20, 0), (30, 1), (30, 0)]
    for blocks, speech in plan:
        for _ in range(blocks):
            x = rng.normal(0, noise_rms, n)
            if speech: x = x + rng.normal(0, speech_rms, n)
            for s in seg.feed(x.astype(np.float32)):
                if s.kind is SegmentKind.FINAL: finals.append(round(len(s.audio) / 16000, 1))
    return finals
for noise, speech in [(0.0015, 0.05), (0.02, 0.15), (0.13, 0.4)]:
    print(f"noise {noise} speech {speech}: final lengths (s) = {run(noise, speech)}  (expect 2 finals of about 2.5 and 3.5 s)")
