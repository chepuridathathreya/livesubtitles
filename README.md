# livesubtitles

Live captions for people who cannot hear, in the languages they actually speak, running on your own machine with no cloud and no cost.

> **Status: early prototype.** Phase 1 (live speech to captions) works. It is not finished, and several parts are unmeasured or unsolved. See [Limits](#limits-read-this-first).

## The problem

Millions of people can read, speak and think perfectly well but cannot hear. In a conversation, a meeting or a family dinner they miss what is being said. Live-caption tools exist, but they usually:

- need an internet connection and send private conversations to a server,
- cost money, or
- work well in English and poorly in Indian languages such as **Telugu** and **Hindi**.

This project aims to turn speech into readable captions, quickly, in **Telugu, Hindi and English**, using free, open models that run locally. The reader sees the captions on their phone.

**Who it is for:** deaf and hard-of-hearing people who read comfortably, and the people who talk with them. A later goal is also sound-event captions (a knock, an alarm), as an accessibility aid. It is **not** a medical device and **not** an alarm system.

### Beyond captions

Because everything runs locally and the speech-to-text engine is a plain Python module, the same engine can later be integrated into **robotics**: a robot or assistant that listens, transcribes and acts on spoken Telugu, Hindi or English without a network connection. That is a possible future direction, not something this repository does today.

## How it works

```
microphone -> segmenter -> speech model -> caption -> phone browser
   (laptop)   finds where    faster-whisper   text      (same Wi-Fi)
              a sentence      (+ Telugu
              starts/ends     fine-tune)
```

- `audio_input.py` captures the mic at 16 kHz in 100 ms blocks.
- `segmenter.py` detects speech, adapts to room noise, and ends long speech at short pauses to keep delay low.
- `transcriber.py` runs [faster-whisper](https://github.com/SYSTRAN/faster-whisper). English and Hindi use `large-v3-turbo`. Telugu uses a Telugu fine-tune, because general Whisper turbo is poor at Telugu (it loops or writes Latin letters).
- `engine.py` ties them together on a background thread and publishes caption events.
- `web_server.py` and `static/index.html` serve a phone-friendly page (Server-Sent Events, standard library only). Telugu and Hindi render correctly in a browser. A terminal cannot shape Telugu well.

Nothing is written to disk and no audio leaves your machine.

## Quick start (Ubuntu, NVIDIA GPU)

Tested on Ubuntu 24.04, Python 3.12, an RTX 3060 6 GB.

```bash
git clone https://github.com/chepuridathathreya/livesubtitles && cd livesubtitles
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
sudo apt install libportaudio2          # microphone support
```

No sudo? Download the package and unpack it locally. `run.sh` and `native_libs.py` look for it in `vendor/portaudio`:

```bash
mkdir -p vendor/debs && cd vendor/debs && apt-get download libportaudio2 && cd ../..
dpkg -x vendor/debs/libportaudio2*.deb vendor/portaudio
```

**Models.** The English/Hindi model (`large-v3-turbo`, about 1.6 GB) downloads automatically on first run. For Telugu, fetch a community fine-tune that is already in faster-whisper format (Apache-2.0, ungated):

```bash
scripts/fetch_model.sh kavithajohn/whisper-telugu-medium-ct2 models/telugu-medium   # 1.5 GB, better
scripts/fetch_model.sh ukta-app/whisper-telugu-ct2 models/telugu-small              # 248 MB, faster
```

**Run.**

```bash
./run.sh --list-devices                 # find your microphone number
./run.sh --device N                     # captions in the terminal
./run.sh --web --device N               # captions on your phone: open the printed URL or scan the QR code
./run.sh --web --device N --te-model models/telugu-small   # faster Telugu, a little less accurate
```

The phone must be on the same Wi-Fi. The URL contains a random key, and only people with it can see your captions. If the phone cannot connect, check the laptop firewall for the port (8000 by default).

**Microphone level.** Very high input gain makes room noise look like speech. If captions lag or never end, lower the capture gain in your sound settings. The segmenter adapts to noise, but clipping hurts accuracy.

Windows also works (`cuda_dlls.py` handles the CUDA libraries), but the current code is only routinely tested on Ubuntu.

## Limits (read this first)

Numbers marked *measured* were measured on one laptop (Ryzen 7 5800H, RTX 3060 6 GB). They use robotic text-to-speech clips as well as live speech by one speaker in a quiet room, so treat them as rough.

- **Needs an NVIDIA GPU today.** *Measured:* on CPU only, turbo and Telugu-medium take about 10 s per sentence, and Telugu-small about 4-5 s. That is too slow for live use. Whisper always processes a fixed 30 s window, so short sentences are not cheap.
- **The laptop must stay near the speaker.** The phone is only a display. Running the whole thing on a phone is not solved; it would need a different (streaming) model and is a separate project.
- **Delay.** *Measured on GPU, simulated continuous Telugu:* median about 1.4 s, max about 2.3 s with Telugu-medium; about 0.7 s median with Telugu-small. English is quicker. The delay shown on the page is the measured sum of waiting, queue and decoding.
- **Telugu accuracy is usable, not perfect.** The Telugu models are community fine-tunes, and their accuracy on conversational speech is *not* benchmarked here. Spoken English words inside Telugu come out in Telugu script. Expect mistakes on names and technical words.
- **Hindi** has had little testing.
- **VRAM.** *Measured:* turbo and Telugu-medium loaded together use about 4.1 GB.
- Whisper-style models can hallucinate on noise. The code drops obvious cases (a lone "you", hums) and collapses loops, but it can still make things up. Do not rely on it for anything safety-critical.

## Tests and measurements

```bash
.venv/bin/python scripts/segmenter_test.py     # sentence-end detection at three noise levels
.venv/bin/python scripts/latency_test.py       # speed and word coverage (needs espeak-ng)
.venv/bin/python scripts/realtime_test.py      # real-time lag through the full engine
```

## Roadmap

Work is split into phases, built in this order. The full architecture, model comparison and pass criteria are in [docs/BLUEPRINT.md](docs/BLUEPRINT.md). Items marked `[verify]` there are unconfirmed.

| Phase | Goal | Status |
|---|---|---|
| 1 | Microphone to live captions (English, Hindi, Telugu) | **Working prototype**; delay and Telugu accuracy still being tuned |
| 5 (early part) | Phone browser page served from the laptop | **Working prototype** (subtitle-style layout, language buttons, honest delay readout) |
| 2 | Sound-event captions (knock, alarm, fan) with a start/continue/stop event tracker | Planned |
| 3 | Translation between Telugu, Hindi and English | Planned |
| 4 | Capture computer audio (video calls, YouTube) | Planned |
| 6 | Phone as the microphone, then Android audio capture | Planned |
| 7 | Optimisation and a tiered, offline mode | Planned |
| **On-device** | Run on a phone with no big GPU and no laptop beside the user | **Research stage.** See [docs/DICTATION-RESEARCH.md](docs/DICTATION-RESEARCH.md) |

Longer term: integrating the speech engine into robotics (a robot that listens in Telugu, Hindi or English, offline).

## Models and licences

Check each model's own licence before relying on it. The two Telugu entries were read from their Hugging Face pages; the Whisper entry is the licence of the original OpenAI Whisper release and was not re-checked on the converted model's page.

| Model | Use | Licence |
|---|---|---|
| `mobiuslabsgmbh/faster-whisper-large-v3-turbo` | English, Hindi | MIT (Whisper's licence, unverified for this conversion) |
| `kavithajohn/whisper-telugu-medium-ct2` (from `vasista22/whisper-telugu-medium`) | Telugu | Apache-2.0 |
| `ukta-app/whisper-telugu-ct2` | Telugu (small) | Apache-2.0 |

Models are downloaded by you from Hugging Face and are not included in this repository.

## Contributing

Issues and pull requests are welcome, especially: real Telugu and Hindi test recordings (with consent), measurements on other hardware, and an on-device path that does not need a big GPU.

## License

No licence has been chosen yet. Until one is added, the default is "all rights reserved".
