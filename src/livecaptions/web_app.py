import secrets
import socket
from pathlib import Path

from .config import AudioConfig, SegmenterConfig, TranscriberConfig
from .engine import CaptionEngine
from .transcriber import WhisperTranscriber
from .web_server import make_server

ROOT = Path(__file__).resolve().parents[2]


def lan_address() -> str:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("10.255.255.255", 1))  # no packet is sent; this only picks the interface
        return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()


def print_qr(url: str) -> None:
    try:
        import qrcode

        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.print_ascii(invert=True)
    except ImportError:
        pass


def run_web(args, ui) -> None:
    cpu = {"device": "cpu", "compute_type": "int8"} if args.cpu else {}
    telugu_path = args.te_model or str(ROOT / "models" / "telugu-medium")

    ui.status(f"loading telugu model {Path(telugu_path).name}")
    telugu = WhisperTranscriber(
        TranscriberConfig(model=telugu_path, language="te", allowed_languages=(), **cpu)
    )
    telugu.warm_up()
    ui.status(f"loading {args.model} for english and hindi")
    general = WhisperTranscriber(
        TranscriberConfig(model=args.model, language="en", allowed_languages=(), **cpu)
    )
    general.warm_up()

    engine = CaptionEngine(
        {"te": telugu, "hi": general.with_language("hi"), "en": general},
        AudioConfig(device=args.device),
        SegmenterConfig(endpoint_silence_ms=args.endpoint_ms, min_rms=args.min_rms),
        partials=not args.no_partials,
        language=args.start_language,
    )
    key = secrets.token_urlsafe(6)
    server = make_server(engine, "0.0.0.0", args.port, key)
    url = f"http://{lan_address()}:{args.port}/?key={key}"
    engine.start()
    ui.status("open this on your phone (same wi-fi), or scan the code")
    print_qr(url)
    print(url, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        ui.status("stopped")
