#!/usr/bin/env bash
# Download DAVIS 2017 480p dataset for background distraction in DistractingCS.
#
# Dataset structure after extraction:
#   <OUT_DIR>/
#     bear/00000.jpg ...
#     boat/00000.jpg ...
#     ...  (60 training + 30 validation video dirs)
#
# distracting_control reads directly from <dataset_path>/<video_name>/<frame>.jpg
# Pass this OUT_DIR as --background-dataset-path in eval.py and run_cloud.sh.
#
# Source: Official DAVIS 2017 480p (trainval), ~2.3 GB compressed.
# Mirrors:
#   1. ETH Zurich (official):
#      https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
#   2. Google Cloud Storage (dm-lab mirror, check availability):
#      gs://dm-lab-datasets/davis.tar.gz
#
# Usage:
#   bash scripts/download_davis.sh                        # → ~/datasets/DAVIS
#   bash scripts/download_davis.sh /path/to/davis_root   # custom location
set -euo pipefail

OUT_DIR="${1:-$HOME/datasets/DAVIS}"
ZIP_URL="https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"
ZIP_FILE="/tmp/DAVIS-2017-trainval-480p.zip"

echo "=== Downloading DAVIS 2017 (480p, ~2.3 GB) ==="
echo "    → $OUT_DIR"

# Download
if [[ -f "$ZIP_FILE" ]]; then
  echo "  Archive already at $ZIP_FILE, skipping download."
else
  echo "  Downloading from ETH Zurich..."
  wget -q --show-progress -O "$ZIP_FILE" "$ZIP_URL" \
    || { echo "wget failed; trying curl..."; curl -L --progress-bar -o "$ZIP_FILE" "$ZIP_URL"; }
fi

# Extract
echo "  Extracting..."
TMPDIR="/tmp/davis_extract_$$"
mkdir -p "$TMPDIR"
unzip -q "$ZIP_FILE" -d "$TMPDIR"

# distracting_control expects: <dataset_path>/<video_name>/<frame>.jpg
# The zip extracts to: DAVIS/JPEGImages/480p/<video_name>/
INNER="$TMPDIR/DAVIS/JPEGImages/480p"
if [[ ! -d "$INNER" ]]; then
  # Some mirrors have a different structure; search for the JPEGImages dir
  INNER=$(find "$TMPDIR" -type d -name "480p" | head -1)
fi

mkdir -p "$OUT_DIR"
# Move each video directory directly into OUT_DIR
for video_dir in "$INNER"/*/; do
  video_name=$(basename "$video_dir")
  if [[ -d "$OUT_DIR/$video_name" ]]; then
    echo "  skipping $video_name (already exists)"
  else
    mv "$video_dir" "$OUT_DIR/"
  fi
done

rm -rf "$TMPDIR"

echo ""
N=$(ls "$OUT_DIR" | wc -l | tr -d ' ')
echo "=== Done: $N video directories in $OUT_DIR ==="
echo "    Use as: --background-dataset-path $OUT_DIR"
