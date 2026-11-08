import argparse

from .audio_input import MicrophoneStream, list_input_devices
from .config import AudioConfig, SegmenterConfig, TranscriberConfig
from .console_ui import ConsoleCaptions
from .segmenter import SegmentKind, UtteranceSegmenter
from .transcriber import WhisperTranscriber
from .web_app import run_web


def parse_device(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="livecaptions")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device", type=parse_device, default=None)
    parser.add_argument("--model", default=TranscriberConfig.model)
    parser.add_argument("--language", default=None, help="force one language, e.g. te")
    parser.add_argument(
        "--languages",
        default=",".join(TranscriberConfig.allowed_languages),
        help="auto-detect among these codes (comma separated), or 'all'",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--web", action="store_true", help="serve live captions to a phone browser")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--te-model", default=None, help="Telugu model folder (default models/telugu-medium)")
    parser.add_argument("--start-language", default="te", choices=["te", "hi", "en"])
    parser.add_argument("--no-partials", action="store_true")
    parser.add_argument("--endpoint-ms", type=int, default=SegmenterConfig.endpoint_silence_ms)
    parser.add_argument("--min-rms", type=float, default=SegmenterConfig.min_rms)
    return parser


def print_devices() -> None:
    for device in list_input_devices():
        print(
            f"{device.index:>3}  {device.name}  [{device.host_api}]"
            f"  {device.channels}ch  {device.default_rate:.0f}Hz"
        )


def run(args: argparse.Namespace) -> None:
    if args.web:
        run_web(args, ConsoleCaptions())
        return
    audio_config = AudioConfig(device=args.device)
    segmenter_config = SegmenterConfig(
        endpoint_silence_ms=args.endpoint_ms, min_rms=args.min_rms
    )
    device_overrides = {"device": "cpu", "compute_type": "int8"} if args.cpu else {}
    transcriber_config = TranscriberConfig(
        model=args.model,
        language=args.language,
        allowed_languages=()
        if args.languages == "all"
        else tuple(code.strip() for code in args.languages.split(",") if code.strip()),
        **device_overrides,
    )

    ui = ConsoleCaptions()
    ui.status(f"loading {transcriber_config.model} on {transcriber_config.device}")
    transcriber = WhisperTranscriber(transcriber_config)
    transcriber.warm_up()
    segmenter = UtteranceSegmenter(audio_config, segmenter_config)

    ui.status("listening. press ctrl+c to stop")
    with MicrophoneStream(audio_config) as mic:
        try:
            while True:
                block = mic.read()
                if block is None:
                    continue
                for segment in segmenter.feed(block):
                    is_final = segment.kind is SegmentKind.FINAL
                    if not is_final and args.no_partials:
                        continue
                    result = transcriber.transcribe(segment.audio, final=is_final)
                    if is_final:
                        ui.final(segment.started_at_s, result)
                    else:
                        ui.partial(segment.started_at_s, result)
                if mic.backlog_ms > 3000:
                    ui.status(f"falling behind: {mic.backlog_ms} ms queued")
        except KeyboardInterrupt:
            for segment in segmenter.flush():
                ui.final(segment.started_at_s, transcriber.transcribe(segment.audio, final=True))
            ui.status(f"stopped. dropped blocks: {mic.dropped_blocks}")


def main() -> None:
    args = build_parser().parse_args()
    if args.list_devices:
        print_devices()
        return
    run(args)


if __name__ == "__main__":
    main()
