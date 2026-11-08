#!/usr/bin/env bash
# usage: fetch.sh <repo> <localdir>   resumable, stall-aware download of every file in a HF repo
repo="$1"; dir="$2"; mkdir -p "$dir"
files=$(curl -s "https://huggingface.co/api/models/$repo" | python3 -c "import sys,json;print('\n'.join(s['rfilename'] for s in json.load(sys.stdin)['siblings'] if not s['rfilename'].startswith('.')))")
for f in $files; do
  mkdir -p "$dir/$(dirname "$f")"
  until curl -fL -C - --connect-timeout 15 --speed-limit 20000 --speed-time 30 -o "$dir/$f" "https://huggingface.co/$repo/resolve/main/$f"; do echo "retry $f"; sleep 2; done
done
echo "ALLDONE $repo"
