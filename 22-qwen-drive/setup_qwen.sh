#!/usr/bin/env bash
# Set up Qwen-Drive-1.0 on a Colab VM (run there, from /content):
#   colab upload -s ep22 setup_qwen.sh /content/setup_qwen.sh
#   colab ssh -s ep22 -- bash /content/setup_qwen.sh
# Own venv (pinned torch 2.8 / transformers 5.14), so it never fights Colab's preinstalled stack
# or Alpamayo's (transformers 5.17). Safe to re-run: every step skips what's already there.
set -euo pipefail
cd /content

QWEN_COMMIT=28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0   # the commit read for episode 22

command -v uv >/dev/null || pip install -q uv

if [ ! -d qwen-drive ]; then
  git clone -q https://github.com/QwenLM/Qwen-Drive-1.0 qwen-drive
fi
git -C qwen-drive checkout -q "$QWEN_COMMIT"

if [ ! -x venv-qwen/bin/python ]; then
  uv venv -q --python 3.12 venv-qwen
fi
export VIRTUAL_ENV=/content/venv-qwen PATH=/content/venv-qwen/bin:$PATH

# torch first, so flash-attn / causal-conv1d pick up their prebuilt wheels for it
uv pip install -q torch==2.8.0 torchvision==0.23.0 setuptools wheel ninja packaging psutil
uv pip install -q --no-build-isolation -r qwen-drive/requirements.txt
uv pip install -q --no-build-isolation --no-deps -e qwen-drive
uv pip install -q pyarrow "huggingface_hub[cli]"

# ~13.7 GB: the VLM plus both planners and the perception head
if [ ! -f weights/Qwen-Drive-1.0-4B/config.json ]; then
  hf download Qwen/Qwen-Drive-1.0-4B --local-dir weights/Qwen-Drive-1.0-4B
fi

python - <<'PY'
import torch, transformers, flash_attn
print("torch", torch.__version__, "cuda", torch.version.cuda, "| transformers", transformers.__version__,
      "| flash_attn", flash_attn.__version__)
print("gpu", torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB")
PY
du -sh weights/Qwen-Drive-1.0-4B
echo "qwen-drive ready"
