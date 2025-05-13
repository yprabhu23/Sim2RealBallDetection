#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="mp4_output"
OUTPUT_DIR="labeled_dataset"
SCRIPT="grounding_dino_mp4_to_coco.py"  # adjust if your script has a different name or path

# make sure output dir exists (the script also does this, but no harm)
mkdir -p "$OUTPUT_DIR/images"

for mp4 in "$INPUT_DIR"/*.mp4; do
  echo "=== Processing: $mp4 ==="
  python "$SCRIPT" \
    -i "$mp4" \
    -o "$OUTPUT_DIR" \
    --prompt "sports ball." \
    --box-threshold 0.25 \
    --text-threshold 0.3
done

echo "All files processed."
