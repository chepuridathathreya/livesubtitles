# How the "microphone button" in browsers and chat apps works

Research notes, 2026-11-08. Question: when you press the mic button in a browser, ChatGPT or Google, text appears fast and clean, and nothing runs on your own GPU. What is happening, and what does it mean for this project?

Evidence labels: **[sourced]** found in a cited page; **[measured]** measured in this repo; **[not established]** I could not confirm it. Search results are secondary sources unless noted, so check the links before relying on any detail.

## 1. Short answer

1. **The heavy work usually happens on the provider's servers, not on your device.** Your machine records audio and sends it; their GPUs or TPUs do the recognition and send text back. That is why your own GPU stays idle.
2. **It feels fast because those systems use streaming recognition.** A streaming model emits words while you are still talking and corrects them as more audio arrives. This repository uses Whisper, which is not a streaming model.
3. **Fully on-device recognition also exists** (Gboard, Android, Chrome language packs), but its Telugu and Hindi availability is not confirmed.

## 2. What each product does

**Chrome and the browser Web Speech API.** Chrome's default implementation captures the microphone and hands the audio to Google's speech service, which returns text to the page. The page never receives the audio. [sourced: [Twilio](https://twilio.com/blog/speech-recognition-browser-web-speech-api), [MDN](https://developer.mozilla.org/docs/Web/API/SpeechRecognition), a [Brave forum post](https://community.brave.com/t/years-of-brave-failing-google-voice-search-with-no-internet-connection-error/395749/2)]. Newer Chrome can run recognition locally using a downloaded language pack (about 60 MB per language per the Blink intent-to-ship thread), via the experimental `processLocally` option and `SpeechRecognition.install()`. [sourced: [MDN install()](https://developer.mozilla.org/docs/Web/API/SpeechRecognition/install_static), [Intent to Ship](https://groups.google.com/a/chromium.org/g/blink-dev/c/VNOok2dbmHM)]

**ChatGPT.** OpenAI's help page says that when you press the microphone, the recorded audio is sent to their models to be transcribed and the text is returned for you to edit. It does not name the model. [sourced: [OpenAI help center](https://help.openai.com/en/articles/12168547-voice-mode-in-work-and-codex)]. A third-party blog says it is Whisper. That claim is not authoritative. [not established: which model the button uses]

**Google (Gboard, Android).** Google's launch write-up describes an all-neural RNN-T that outputs characters and has no attention mechanism. The encoder has 8 LSTM layers, and the model is 450 MB before compression and **80 MB after**. Quantisation gave about 4x smaller and 4x faster, it runs faster than real time on a single phone core, and it replaces the roughly 2 GB search graph of traditional server recognisers. It first shipped on Pixel phones in American English only, and Google says it matched server-model accuracy. It gives no numeric latency. [sourced: [Google Research, "An All-Neural On-Device Speech Recognizer"](https://research.google/blog/an-all-neural-on-device-speech-recognizer/)]. Later reports say Gboard moved on-device in 2019 and emits text as you speak. Android has an on-device recogniser API from API level 31 (`createOnDeviceSpeechRecognizer`), and apps must check `isOnDeviceRecognitionAvailable()` first. [sourced: [Neurohive](https://neurohive.io/en/news/google-s-gboard-now-uses-an-on-device-neural-network-for-speech-recognition/), [Hackernoon test](https://hackernoon.com/testing-androids-on-device-speech-recognition-with-hindi-english-code-switching)]. A developer reported Hindi (`hi-IN`) working offline. Offline Telugu on the system recogniser is [not established].

## 3. Why streaming feels fast and Whisper does not

- **RNN-T** (Recurrent Neural Network Transducer) is frame-synchronous: it can output text as audio arrives, without waiting for the end of the utterance. Attention-based models need the full clip first. Partial text can be revised as more audio arrives, and the model can learn the end-of-speech decision itself. [sourced: [arXiv 2003.12710](https://arxiv.org/pdf/2003.12710v1.pdf), [arXiv 2402.13076](https://arxiv.org/pdf/2402.13076)]
- **Whisper** processes a fixed 30-second window per call. We re-decode the growing sentence for drafts and wait for silence to finalise. **[measured]** on this repo's GPU setup: about 1.4 s median lag for Telugu-medium, and on CPU only about 10 s for turbo and Telugu-medium and 4-5 s for Telugu-small. See `scripts/realtime_test.py` and `scripts/latency_test.py`.

## 4. Why dictation sometimes writes Telugu in English letters

A recogniser chooses a language and script before it writes. With the language set to English (or auto-detect settling on English), Telugu speech is spelled phonetically in Latin letters. Forced to Telugu, a weak general model can do the same: **[measured]** Whisper turbo forced to `te` produced `Iroju mata marangan cala badundi`. This is an explanation from general behaviour plus our own test, not a fact about any specific product's internals.

## 5. Leads for an on-device, no-big-GPU path

None of these is verified by us yet.

- **AI4Bharat IndicConformer** (600M parameters, MIT licence, Telugu listed among 22 languages; the repository requires accepting conditions on Hugging Face). Model page: [indic-conformer-600m-multilingual](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual). Its page also points to a newer model called Indic-Transcribe.
- **Projects that claim to run IndicConformer fully offline on Android through sherpa-onnx**, covering Telugu and Hindi, with a per-language pack around 198 MB under MIT. These are the projects' own claims, from the search snippet only. [sourced, unverified: [iTantra packs](https://huggingface.co/datasets/Mr66/itantra-packs), [Uktam](https://github.com/ashb155/uktam)]
- **sherpa-onnx** lists streaming Zipformer models for several languages, but none for Telugu or Hindi were found. [sourced: [sherpa-onnx](https://pkg.go.dev/github.com/csukuangfj/sherpa-onnx@v1.12.33)]. Per-language Whisper fine-tunes in ONNX exist but are not streaming ([ippocode/indic-asr-onnx](https://huggingface.co/ippocode/indic-asr-onnx)).

## 6. What this means here, and next experiments

- A phone-only version most likely needs a **non-Whisper** model that is either streaming or cheap per sentence. IndicConformer through sherpa-onnx is the first candidate to test.
- Experiments, in order: (1) run an IndicConformer pack on this laptop's CPU with Telugu speech and measure delay and word accuracy against the Telugu Whisper fine-tune; (2) if it is fast enough, test on a mid-range Android phone; (3) check the Android system recogniser's offline Telugu and Hindi availability on the target phone; (4) compare licences and download terms before shipping anything.
- Until then, this repository's working path remains: laptop with an NVIDIA GPU, phone as display.
