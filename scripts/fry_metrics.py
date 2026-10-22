"""Shared helpers: ffmpeg conversion and creak (vocal fry) measurement via CreaPy."""
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "VQVC" / "creapy"))

import creapy  # noqa: E402  (bundled with VQVC: Paierl et al. 2023 creak detector)
import imageio_ffmpeg  # noqa: E402

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
CREAK_THRESHOLD = 0.75  # CreaPy's default decision threshold
HOP_S = 0.010           # CreaPy hop size


def to_wav(src, dst, sr=16000, start=None, duration=None):
    cmd = [FFMPEG, "-y", "-loglevel", "error"]
    if start is not None:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(src)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(dst)]
    subprocess.run(cmd, check=True)
    return Path(dst)


def creak_track(wav_path, gender_model="all"):
    """Per-10ms-frame creak probability and a mask of frames CreaPy treats as voiced speech."""
    X, y_pred, _ = creapy.process_file(str(wav_path), gender_model=gender_model)
    voiced = X["hnr"].notna().to_numpy() if "hnr" in X else y_pred > 0
    return np.asarray(y_pred), voiced


def creak_percent(wav_path, gender_model="all"):
    y, voiced = creak_track(wav_path, gender_model)
    n_voiced = int(voiced.sum())
    if n_voiced == 0:
        return 0.0, 0.0
    creaky = (y >= CREAK_THRESHOLD) & voiced
    return 100.0 * creaky.sum() / n_voiced, 100.0 * n_voiced / len(y)
