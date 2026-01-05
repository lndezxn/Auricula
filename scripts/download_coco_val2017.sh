#!/usr/bin/env bash
set -euo pipefail

# Download COCO 2017 val images and annotations if missing.
# Usage: ./download_coco_val2017.sh [COCO_DIR]
# Default COCO_DIR: assets/coco2017

COCO_DIR=${1:-assets/coco2017}
VAL_URL="http://images.cocodataset.org/zips/val2017.zip"
ANN_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

IMAGES_DIR="$COCO_DIR/images/val2017"
ANN_DIR="$COCO_DIR/annotations"
VAL_ZIP="$COCO_DIR/val2017.zip"
ANN_ZIP="$COCO_DIR/annotations_trainval2017.zip"

mkdir -p "$COCO_DIR"

fetch_zip() {
  local url="$1"
  local zip_path="$2"
  local target_dir="$3"
  if [[ -d "$target_dir" ]]; then
    echo "[skip] $target_dir already exists"
    return 0
  fi
  if [[ ! -f "$zip_path" ]]; then
    echo "[download] $url -> $zip_path"
    curl -L "$url" -o "$zip_path"
  else
    echo "[reuse] $zip_path already downloaded"
  fi
  echo "[unzip] $zip_path"
  unzip -q "$zip_path" -d "$COCO_DIR"
}

fetch_zip "$VAL_URL" "$VAL_ZIP" "$IMAGES_DIR"
fetch_zip "$ANN_URL" "$ANN_ZIP" "$ANN_DIR"

# Show resulting layout
echo "\n[done] COCO 2017 val assets prepared under $COCO_DIR"
find "$COCO_DIR" -maxdepth 2 -type d | sort

echo "\nNext steps:"
echo "- val images: $IMAGES_DIR"
echo "- annotations: $ANN_DIR"
