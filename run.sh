#!/usr/bin/env bash
# Linux launcher: points the loader at the pip-installed CUDA libs and the local PortAudio.
# Usage: ./run.sh --list-devices | ./run.sh --device N [--language te] [--cpu]
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SP="$HERE/.venv/lib/python3.12/site-packages"
LIBS="$HERE/vendor/portaudio/usr/lib/x86_64-linux-gnu"
for d in "$SP"/nvidia/*/lib; do LIBS="$LIBS:$d"; done
export LD_LIBRARY_PATH="$LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$HERE/.venv/bin/python" -m livecaptions "$@"
