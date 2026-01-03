#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${1:?Usage: $0 <repo_id> <out_dir> [revision]}"
OUT_DIR="${2:?Usage: $0 <repo_id> <out_dir> [revision]}"
REVISION="${3:-main}"

mkdir -p "$OUT_DIR"
TASK_FILE="$OUT_DIR/aria2_tasks.txt"

echo "[1/4] Listing files from HF repo: $REPO_ID (rev=$REVISION)"
python - <<'PY'
import os
from huggingface_hub import HfApi

repo_id = os.environ["REPO_ID"]
rev = os.environ["REVISION"]
out_dir = os.environ["OUT_DIR"]
task_file = os.environ["TASK_FILE"]

api = HfApi()
files = api.list_repo_files(repo_id=repo_id, revision=rev)

def url_for(path):
    return f"https://huggingface.co/{repo_id}/resolve/{rev}/{path}?download=true"

lines = []
for p in files:
    d = os.path.join(out_dir, os.path.dirname(p))
    out = os.path.basename(p)
    lines.append(url_for(p))
    lines.append(f"  dir={d}")
    lines.append(f"  out={out}")
    # aria2 支持 per-item header，但我们统一在命令行传 --header，更简单
    lines.append("")

with open(task_file, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Wrote {len(files)} files into {task_file}")
PY
echo "[2/4] aria2c downloading (resume enabled). Output: $OUT_DIR"
echo "      Task file: $TASK_FILE"

HEADER_ARGS=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  HEADER_ARGS+=(--header="Authorization: Bearer ${HF_TOKEN}")
fi

aria2c -c -i "$TASK_FILE" \
  -x 8 -s 8 -j 1 -k 1M \
  --max-tries=0 --retry-wait=5 \
  --connect-timeout=30 --timeout=30 \
  --allow-overwrite=true --auto-file-renaming=false \
  "${HEADER_ARGS[@]}"

echo "[3/4] Basic file presence check"
test -f "$OUT_DIR/config.json" || (echo "Missing config.json in $OUT_DIR" && exit 2)

echo "[4/4] Offline load check with transformers (no network)"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

python - <<'PY'
import os
from transformers import AutoProcessor

local_dir = os.environ["OUT_DIR"]

proc = AutoProcessor.from_pretrained(local_dir, local_files_only=True)
print("AutoProcessor OK")

# 这里不强依赖具体类，避免你没装对应模型类时报错；
# 你后续在项目里会用 Qwen2VLForConditionalGeneration.from_pretrained(local_dir, ...)
print("Local directory is ready for from_pretrained(local_dir).")
PY

echo "DONE: Model repo is downloaded to: $OUT_DIR"
echo "Tip: re-run the same command to resume if interrupted."