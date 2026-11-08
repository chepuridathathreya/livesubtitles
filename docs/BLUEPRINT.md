# Live Subtitles: Technical Blueprint

Written 2026-11-08. Evidence levels used below:
- **[known]** stable facts I'm confident of from model cards and official docs.
- **[verify]** from memory; confirm in a spike before depending on it.
- **[measure]** must be benchmarked on your RTX 3060 6 GB. No numbers are invented here.

Web research was not run in this session. Every [verify] item needs a check against the official repo or model card before it becomes a decision.

## A. Architecture

```
SYSTEM A sources            SYSTEM B sources
 mic / USB mic               Windows WASAPI loopback
 phone-as-mic (later)        Android playback capture (later)
                             TV via optical/3.5mm/HDMI-extractor -> USB interface
                             mic near TV (fallback)
          \                         /
           +--> Audio Input Manager (SHARED) <--+
                 device select, resample to 16 kHz mono float32,
                 100 ms blocks, ring buffer, interruption handling
                          |
                 Preprocessing (SHARED)
                 normalize, optional denoise, per-model windowing
                          |
        +-----------------+------------------+
        | Branch A: speech                   | Branch B: sound events
        | endpointing/VAD                    | tagger (multi-label, ~1 s hop)
        | ASR + language ID                  | per-class smoothing
        | translation (separate module)      | EVENT TRACKER state machine
        +-----------------+------------------+
                          |
                 Caption Manager (SHARED)
                 merge, dedupe, priority, timestamps, expiry
                          |
                 Transport: FastAPI + WebSocket (Phase 5)
                          |
                 UI: web page, later Android and TV
```

Shared: input manager, preprocessing, caption manager, transport, UI.
System A only: mic selection, sound-event branch priority, echo and room-noise handling.
System B only: loopback and capture backends, subtitle-stream reader, A/V sync offset, DRM-aware error states.

Each stage is an interface (`AudioSource`, `Transcriber`, `SoundTagger`, `Translator`, `CaptionSink`). A model is a plugin behind its interface.

**Classification is not understanding.** A tagger answers "what does this 1 s window sound like". The event tracker turns that into *started / continuing / stopped*:
1. Smooth each class probability over time (EMA or median).
2. Hysteresis: start when `p > on_threshold` for N windows, stop when `p < off_threshold` for M windows. `off < on` prevents flicker.
3. State per class: `IDLE -> STARTED -> ONGOING -> STOPPED`. Emit one caption per transition, not per window.
4. Class families: continuous (fan, rain, wind, vehicle) use start/stop; impulsive (door knock, glass break, clap, cough) emit a single event with a cooldown.
5. Priority: alarms, glass, knocks, speech directed at user outrank ambient hum.

## B. Model comparison

### Speech

| Option | License | Notes |
|---|---|---|
| Whisper (OpenAI) | MIT [known] | Multilingual, language ID built in, 30 s window design, not natively streaming. |
| faster-whisper (CTranslate2) | MIT [known] | Same weights, much faster, lower VRAM, int8/fp16. Recommended runtime. |
| large-v3-turbo | MIT [known] | Pruned decoder, about 809M params [verify], fp16 weights about 1.6 GB [verify]. Fits 6 GB VRAM. |
| distil-whisper | MIT [verify] | English-focused. Not useful for Telugu or Hindi. |
| whisper.cpp | MIT [known] | Good CPU and Vulkan story, less convenient from Python. |

Streaming strategy for Phase 1: energy endpointing, re-transcribe the growing utterance every second for partials, finalize on 700 ms silence. This is simpler than a full local-agreement algorithm. Upgrade path: Silero VAD, then LocalAgreement-style stable prefixes.

Known weakness [verify]: Whisper accuracy on Telugu is clearly below English and Hindi. Measure with your own Telugu samples before promising Telugu captions.

### Sound events

| Model | License | Size | Realtime on your laptop | Fit |
|---|---|---|---|---|
| YAMNet | Apache-2.0 [known] | about 3.7 MB, 521 AudioSet classes [known] | Yes, CPU is enough [verify] | Best first baseline. Tiny, 0.96 s frames. Coarse accuracy. |
| PANNs (CNN14 etc.) | MIT repo [verify] | about 80 MB for CNN14 [verify] | Likely yes on GPU [measure] | Stronger tagger, 527 classes. Needs PyTorch. |
| AST | BSD-3 weights, MIT code [verify] | about 87M params [verify] | Probably heavier, 10 s input design [measure] | Higher accuracy, worse latency. Phase 2 comparison only. |
| CLAP (LAION) | Apache-2.0 [verify] | few hundred MB [verify] | Possible [measure] | Zero-shot by text prompt: add "door knock" without retraining. Slower, less calibrated. |
| Qwen-Audio / Qwen2-Audio | Apache-2.0 [verify] | 7B [verify] | **No.** Does not fit 6 GB at useful speed. | Offline "describe this clip" only, or cloud. Not a real-time tagger. |

Recommended combination for your hardware:
- ASR: faster-whisper `large-v3-turbo` fp16 on GPU, `small` int8 on CPU as fallback.
- Sound events: YAMNet first (ONNX or TFLite, no torch), compare against PANNs in the Phase 2 spike.
- CLAP as an optional "custom sound" add-on later.
- Do not run all models on the GPU at once without measuring VRAM. Whisper turbo plus a small tagger should fit; measure.

Concurrent GPU use is the main risk: ASR bursts will steal time from the tagger. Mitigation: the tagger runs on CPU.

### Translation

| Option | Offline | Telugu / Hindi | Notes |
|---|---|---|---|
| NLLB-200 (Meta) | Yes | Yes [known, 200 languages] | License is CC-BY-NC for weights [verify]. Check before any commercial use. 600M distilled variant is feasible on your GPU. |
| IndicTrans2 (AI4Bharat) | Yes | Yes, built for Indic [known] | MIT [verify]. Likely the best open Indic quality. Setup is heavier. |
| Cloud (Google/DeepL/Azure) | No | Yes | Quality and speed good, cost per character, privacy tradeoff. |
| LLM via OpenRouter | No | Yes | Flexible, context-aware, higher latency. |

Translation sits behind a `Translator` interface so you can switch between IndicTrans2, NLLB, and cloud without touching ASR.

## C. Audio capture

**Microphone.** PortAudio via `sounddevice`. Open at 16 kHz mono; if the device refuses, open at native rate and resample. WASAPI shared mode, MME and DirectSound behave differently, so `--list-devices` shows the host API.

**Windows system audio.** WASAPI loopback captures what the default output device plays [known]. In Python: `pyaudiowpatch` (a PyAudio fork with loopback) or `soundcard` [verify], or FFmpeg's `wasapi` input [verify]. Known gotchas: loopback silence when nothing plays (no samples are delivered, so handle gaps), sample rate follows the output device, and protected content may render silent or blocked in some apps [verify per app]. This is the path for YouTube and local players in Phase 4.

**Android.** `AudioPlaybackCapture` (API 29+) lets an app capture other apps' audio after MediaProjection consent [known]. Apps can opt out via capture policy, and DRM/streaming apps commonly do [known in general, verify per app]. Many streaming apps will return silence. The Poco X6 will tell us the real situation empirically. The phone can also act as a remote microphone streaming to the laptop, which avoids the capture restrictions entirely.

**Smart TV.** Apps on Android TV / Google TV can run, but the same playback-capture and DRM limits apply, and capturing other apps' audio is generally not available to third-party TV apps [verify]. Samsung Tizen and LG webOS are closed ecosystems with their own SDKs [verify]. Practical TV route: the laptop or phone does the processing, TV audio arrives via hardware (below) or microphone.

**External hardware.** Options in order of simplicity:
1. TV optical (TOSLINK) or 3.5 mm headphone out into a USB audio interface or optical-to-USB adapter. Cheap, no HDCP involved.
2. HDMI audio extractor into a USB capture interface. HDCP-protected sources may refuse to output unprotected audio, or the extractor must be an ARC/eARC-compliant path [verify]. Compatibility varies by TV and extractor. We are **not** circumventing HDCP or DRM; if a device refuses, the answer is "unsupported", not a workaround.
3. Latency: expect 20-100 ms of capture latency [measure], negligible against ASR.

**Microphone near the TV.** Works everywhere, worst quality. Expect room echo and overlap with the user's own voice. Useful as the universal fallback.

**Subtitle streams.** Read existing subtitles where legitimately accessible (local `.srt`/`.vtt`, ffprobe on local files, player APIs). Not available for most streaming apps; never assume them. Subtitles skip ASR but still need translation and sync.

**Platform limits summary:** DRM-protected playback may be uncapturable on any platform. The app must show a clear "source is silent or protected" state rather than hang or guess.

## D. Technology selection

Use now:
- Python 3.12 venv. System Python is 3.14; ML wheels for 3.14 are not guaranteed, so 3.12 avoids install failures.
- `faster-whisper`, `sounddevice`, `numpy`.
- `nvidia-cublas-cu12` + `nvidia-cudnn-cu12` pip wheels so CTranslate2 can run on CUDA without a system CUDA toolkit. Roughly 1 GB of disk.
- FFmpeg (already installed) for file-based tests and later media input.

Add later:
- Phase 2: `onnxruntime` for YAMNet, `torch` only if PANNs/AST wins (large install; you have about 8 GB free on D:).
- Phase 3: `ctranslate2` Marian/NLLB or IndicTrans2.
- Phase 4: `pyaudiowpatch` or equivalent loopback.
- Phase 5: FastAPI, uvicorn, WebSockets, SQLite (stdlib `sqlite3`).

Avoid for now: Docker, Redis or queues, any frontend framework, Raspberry Pi/ESP32, Qwen-Audio, multi-process orchestration, a database before there is something to store.

Disk: free space on D: is about 8 GB. Budget: venv about 2 GB, large-v3-turbo about 1.6 GB, torch later about 3 GB. Watch it.

AI tooling: keep model/API calls behind the interfaces above, read keys from environment variables, and keep an OpenAI-compatible client for OpenRouter-style cloud paths. Nothing in the engine imports a provider SDK directly.

## E. Roadmap and exit criteria

| Phase | Result | Pass criteria |
|---|---|---|
| 1 Mic + live ASR | Console captions with timestamps, language tag, latency | Final caption within 1.5 s of end of utterance on GPU [measure]. No caption on 60 s of silence or fan noise. Mic device selectable. |
| 2 Sound events | YAMNet baseline, then compare PANNs/AST/CLAP; event tracker | Labeled test set of about 10 clips per target sound recorded by you. Report precision and recall per class. Fan yields exactly one start and one stop over a run, not repeats. |
| 3 Translation | EN->TE, HI->EN, TE->HI captions | Native-speaker review of 30 sentences per pair. Added latency under 500 ms per utterance [measure]. |
| 4 Windows loopback | YouTube and local video captioned | Works with local file and browser YouTube. Documented behavior for protected apps. |
| 5 Backend + web UI | FastAPI WebSocket, responsive page | Caption visible in a phone browser on the same Wi-Fi under 2 s end to end [measure]. |
| 6 Cross-device | Phone as mic, laptop as server, then Android capture spike | Poco X6 streams mic audio to the laptop. Android playback-capture feasibility documented per app. |
| 7 Optimization | CPU/GPU/RAM budget, offline mode | Targets set from Phase 1-5 measurements, not guessed. |

Hardware: nothing to buy now. Laptop plus phone covers Phases 1-6. Buy later only with a clear reason, for example an optical-to-USB interface (low cost) if TV audio capture is needed.

## UI plan (from the reference video, UI only)

The reference's visual language, adapted to captions and sound events. The reference's video-classification concept is ignored.

- Black background, monospace, uppercase labels, tiny type for metadata, high contrast.
- **Headline panel**: one line of plain-language description of what is happening now (like the reference's "SITTING UPRIGHT THROUGHOUT..." line), with a muted "LISTENING..." state while idle.
- **Stats row**: `TIME`, `EVENTS`, `LANGUAGE`, and later `COST` for cloud use. Always shows `LOCAL` or `CLOUD` for privacy transparency.
- **Ranked class list**: every tracked sound shown as a row with percent confidence, top row highlighted white, alert-class rows highlighted red (glass, alarm, knock). Rows below a floor collapse to 0.0%.
- **Latency footer**: `INFERENCE / 226 MS` style, our measured pipeline latency.
- **Caption overlay**: large readable subtitle text, bottom-centered, over the live view when video is present. Size, font and contrast adjustable.
- Events are shown as a timeline of `STARTED / ONGOING / STOPPED` rows with timestamps.

The terminal in Phase 1 already uses this style (uppercase status lines, per-caption latency footer). The web UI in Phase 5 implements it fully.

## Privacy and safety rules (apply from Phase 1)

- No audio is written to disk by default.
- Local processing is the default; any cloud call is announced in the UI.
- Low-confidence output is suppressed or marked uncertain. The Whisper wrapper already drops segments that look like silence hallucinations.
- Accessibility aid, not a medical device and not a hearing-aid replacement. Do not use it as an alarm system.
