"""Compare VQVC outputs against the original clip: fry, voice identity, intelligibility, quality, sync."""
import argparse
import json
import re
from pathlib import Path

import jiwer
import librosa
import numpy as np
import parselmouth
import torch
import torchaudio

from fry_metrics import ROOT, creak_percent

SR = 16000


def load(path):
    return librosa.load(path, sr=SR, mono=True)[0]


def median_f0(wav, floor, ceiling):
    pitch = parselmouth.Sound(wav, SR).to_pitch(time_step=0.01, pitch_floor=floor, pitch_ceiling=ceiling)
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]
    return float(np.median(f0)) if len(f0) else float("nan")


def envelope(wav, hop=160):
    return librosa.feature.rms(y=wav, frame_length=640, hop_length=hop)[0]


def lag_ms(a, b, max_lag_frames=50):
    """Lag of b relative to a (positive = b late), from cross-correlating 10 ms loudness envelopes."""
    ea, eb = envelope(a), envelope(b)
    n = min(len(ea), len(eb))
    ea, eb = ea[:n] - ea[:n].mean(), eb[:n] - eb[:n].mean()
    lags = range(-max_lag_frames, max_lag_frames + 1)
    scores = [np.dot(ea[max(0, -k):n - max(0, k)], eb[max(0, k):n - max(0, -k)]) for k in lags]
    return 10.0 * lags[int(np.argmax(scores))]


class Speaker:
    def __init__(self):
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "models" / "ecapa"),
            local_strategy=LocalStrategy.COPY)

    def embed(self, wav):
        with torch.no_grad():
            return self.model.encode_batch(torch.from_numpy(wav).unsqueeze(0)).squeeze()

    def similarity(self, a, b):
        return float(torch.nn.functional.cosine_similarity(self.embed(a), self.embed(b), dim=0))


class Quality:
    """torchaudio SQUIM: reference-free estimate of PESQ (1-4.5, higher = cleaner/less distorted)."""

    def __init__(self):
        self.model = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model()

    def pesq(self, wav, chunk_s=10):
        scores = []
        for i in range(0, len(wav) - SR * 3, SR * chunk_s):
            with torch.no_grad():
                _, pesq, _ = self.model(torch.from_numpy(wav[i:i + SR * chunk_s]).unsqueeze(0))
            scores.append(float(pesq))
        return float(np.mean(scores))


class Transcriber:
    def __init__(self):
        from faster_whisper import WhisperModel
        self.model = WhisperModel("small.en", device="cpu", compute_type="int8",
                                  download_root=str(ROOT / "models" / "whisper"))

    def __call__(self, wav):
        segments, _ = self.model.transcribe(wav, language="en", beam_size=5)
        return " ".join(s.text for s in segments).strip()


def norm(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]", " ", text.lower())).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("original", help="16 kHz original clip")
    ap.add_argument("reference", help="same speaker, different audio")
    ap.add_argument("other_speaker", help="a different speaker, to calibrate the similarity scale")
    ap.add_argument("outdir", help="folder with VQVC outputs and convert_report.json")
    ap.add_argument("--gender", choices=["male", "female"], required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    conv = json.loads((outdir / "convert_report.json").read_text())
    floor, ceiling = (40, 300) if args.gender == "male" else (60, 500)

    orig, ref, other = load(args.original), load(args.reference), load(args.other_speaker)
    spk, qual, asr = Speaker(), Quality(), Transcriber()
    orig_text = asr(orig)

    rows = {}
    for name, path in [("original", Path(args.original))] + [(n, outdir / f"{n}.wav") for n in conv["outputs"]]:
        wav = load(path)
        creak, voiced = creak_percent(path, gender_model=args.gender)
        text = orig_text if name == "original" else asr(wav)
        rows[name] = {
            "creak_pct": round(creak, 1),
            "voiced_pct": round(voiced, 1),
            "median_f0_hz": round(median_f0(wav, floor, ceiling), 1),
            "speaker_sim_to_reference": round(spk.similarity(wav, ref), 3),
            "wer_vs_original_pct": round(100 * jiwer.wer(norm(orig_text), norm(text)), 1),
            "pesq_estimate": round(qual.pesq(wav), 2),
            "length_diff_ms": 0.0 if name == "original" else round(conv["outputs"][name]["diff_ms"], 1),
            "lag_first_5s_ms": 0.0 if name == "original" else lag_ms(orig[:5 * SR], wav[:5 * SR]),
            "lag_last_5s_ms": 0.0 if name == "original" else lag_ms(orig[-5 * SR:], wav[-5 * SR:]),
            "transcript": text,
        }
        print(name, {k: v for k, v in rows[name].items() if k != "transcript"}, flush=True)

    calib = {"same_speaker_original_vs_reference": round(spk.similarity(orig, ref), 3),
             "different_speaker_vs_reference": round(spk.similarity(other, ref), 3)}
    print("calibration", calib)
    (outdir / "eval_report.json").write_text(json.dumps({"rows": rows, "calibration": calib}, indent=2))


if __name__ == "__main__":
    main()
