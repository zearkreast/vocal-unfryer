#!/usr/bin/env bash
# Reproduce the VoiceQualityVC test setup. Run from the repo root.
set -euo pipefail

VQVC_COMMIT=40923058ee1893a5d0774ad1edeb5c237e869b7a

if [ ! -d VQVC ]; then
  git clone -q https://github.com/Hfkml/VQVC.git
  git -C VQVC checkout -q "$VQVC_COMMIT"
fi

# VQVC imports wavlm/ but .gitignores it: use Microsoft's code, with a package-relative import
# so it doesn't pick up VQVC's own modules.py.
mkdir -p VQVC/wavlm
curl -sfL -o VQVC/wavlm/WavLM.py https://raw.githubusercontent.com/microsoft/unilm/master/wavlm/WavLM.py
curl -sfL -o VQVC/wavlm/modules.py https://raw.githubusercontent.com/microsoft/unilm/master/wavlm/modules.py
sed -i 's/^from modules import (/from .modules import (/' VQVC/wavlm/WavLM.py
printf 'from .WavLM import WavLM, WavLMConfig\n' > VQVC/wavlm/__init__.py

# Model weights: VQVC generator (~440 MB) and WavLM-Large (~1.26 GB, s3prl mirror).
mkdir -p downloads/vqvc_release downloads/wavlm_s3prl
[ -f downloads/vqvc_release/VQVC.pth ] || \
  curl -sfL https://github.com/Hfkml/VQVC/releases/download/v1.0.0/G_VQVC.tar.gz | tar -xz -C downloads/vqvc_release
[ -f downloads/wavlm_s3prl/wavlm_large.pt ] || \
  curl -sfL -o downloads/wavlm_s3prl/wavlm_large.pt https://huggingface.co/s3prl/converted_ckpts/resolve/main/wavlm_large.pt

echo "Setup done. Pass --wavlm ../downloads/wavlm_s3prl/wavlm_large.pt to scripts/vqvc_convert.py."
