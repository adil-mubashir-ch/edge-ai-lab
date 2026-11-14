#!/usr/bin/env bash
# Set up NVIDIA Alpamayo 2 Edge on a Colab VM (run there, from /content):
#   colab upload -s ep22 setup_alpamayo.sh /content/setup_alpamayo.sh
#   colab ssh -s ep22 -- bash /content/setup_alpamayo.sh
# The weights and the PhysicalAI-AV dataset are gated: accept both on Hugging Face first, then log the
# VM in yourself (`colab ssh -s ep22`, then `hf auth login`). Own venv from the repo's lockfile
# (transformers 5.17), separate from Qwen-Drive's. Safe to re-run.
set -euo pipefail
cd /content

A2E_COMMIT=02f55a7aeabc1a2b9f7a80445c08e40bddc89f5e    # the commit read for episode 22
A2E_REVISION=f8951d095de3589b3face1dec3f4394c8ecedb38  # the README's pinned checkpoint

command -v uv >/dev/null || pip install -q uv

if [ ! -d alpamayo2-edge ]; then
  git clone -q https://github.com/NVlabs/alpamayo2-edge alpamayo2-edge
fi
git -C alpamayo2-edge checkout -q "$A2E_COMMIT"

export UV_PROJECT_ENVIRONMENT=/content/venv-a2e
(cd alpamayo2-edge && uv sync -q --locked --extra dataset --extra notebook)
/content/venv-a2e/bin/python -m pip --version >/dev/null 2>&1 || uv pip install -q --python /content/venv-a2e/bin/python pip
# FlashAttention 2 is optional: the default path is SDPA, which is what we time first
uv pip install -q --python /content/venv-a2e/bin/python --no-build-isolation flash-attn==2.8.3 \
  || echo "flash-attn unavailable: SDPA only"

if [ ! -f weights/Alpamayo2-Edge/config.json ]; then
  /content/venv-a2e/bin/python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="nvidia/Alpamayo2-Edge", revision="$A2E_REVISION",
                  local_dir="/content/weights/Alpamayo2-Edge")
PY
fi
du -sh weights/Alpamayo2-Edge
echo "alpamayo2-edge ready"
