"""Run VoiceQualityVC on one clip with several voice-quality presets.

Feature values follow the VQVC notebook's scale (roughly -3..3, 0 = average voice):
its "creaky" target is creak=3, cpps=-1, h1h2=-2, h1a3=-2.
"""
import argparse
import json
import logging
import sys
import types
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
VQVC = ROOT / "VQVC"
sys.path.insert(0, str(VQVC))
# utils.py imports a vocoder package the repo doesn't ship and torchvision; neither is used for inference.
for name in ("hifigan", "torchvision"):
    sys.modules.setdefault(name, types.ModuleType(name))

import utils  # noqa: E402
from models import SynthesizerTrn  # noqa: E402
from speaker_encoder.voice_encoder import SpeakerEncoder  # noqa: E402
from wavlm import WavLM, WavLMConfig  # noqa: E402

logging.getLogger().setLevel(logging.WARNING)  # utils.py sets the root logger to DEBUG on import

SR, HOP = 16000, 320

# name: (creak, cpps, h1h2, h1a3, pitch_var); pitch level comes from --pitch
PRESETS = {
    "neutral":       (0.0, 0.0, 0.0, 0.0, 0.0),
    "less_fry":      (-1.0, 0.5, 0.5, 0.0, 0.0),
    "much_less_fry": (-2.0, 1.0, 1.0, 0.0, 0.0),
    "more_fry":      (3.0, -1.0, -2.0, -2.0, 0.0),  # sanity check: should sound creakier
}


def load_models(device, wavlm_path, vqvc_path):
    hps = utils.get_hparams_from_file(str(VQVC / "configs" / "freevc.json"))
    net_g = SynthesizerTrn(hps.data.filter_length // 2 + 1,
                           hps.train.segment_size // hps.data.hop_length, **hps.model)
    utils.load_checkpoint(str(vqvc_path), net_g, optimizer=None, strict=True)
    net_g.eval().to(device)

    ckpt = torch.load(wavlm_path, map_location="cpu", weights_only=True)
    cmodel = WavLM(WavLMConfig(ckpt["cfg"]))
    cmodel.load_state_dict(ckpt["model"])
    cmodel.eval().to(device)

    smodel = SpeakerEncoder(str(VQVC / "speaker_encoder" / "ckpt" / "pretrained_bak_5805000.pt"),
                            device=device)
    return net_g, cmodel, smodel


@torch.no_grad()
def convert(net_g, c, g, values, pitch, device):
    creak, cpps, h1h2, h1a3, pitch_var = values
    # The conditioning length is one frame shorter than the content for some inputs; the notebook tries both.
    for dim in (c.shape[-1], c.shape[-1] - 1):
        base = torch.zeros((1, 1, dim), device=device)
        try:
            out = net_g.infer(c, g=g, creaks=base + creak, cpps=base + cpps, h1h2s=base + h1h2,
                              pitches=base + pitch, h1a3s=base + h1a3, pitch_vars=base + pitch_var)
            return out[0][0].float().cpu().numpy()
        except RuntimeError:
            continue
    raise RuntimeError("conversion failed for both conditioning lengths")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="clip to convert (any sample rate; resampled to 16 kHz)")
    ap.add_argument("reference", help="same speaker, ideally 30 s+ with little fry")
    ap.add_argument("outdir")
    ap.add_argument("--pitch", type=float, default=0.0, help="VQVC notebook: about -1 male, +1 female")
    ap.add_argument("--presets", default=",".join(PRESETS))
    ap.add_argument("--wavlm", default=str(ROOT / "downloads" / "wavlm" / "WavLM-Large.pt"))
    ap.add_argument("--vqvc", default=str(ROOT / "downloads" / "vqvc_release" / "VQVC.pth"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    net_g, cmodel, smodel = load_models(device, args.wavlm, args.vqvc)

    ref, _ = librosa.load(args.reference, sr=SR)
    ref, _ = librosa.effects.trim(ref, top_db=20)
    g = torch.from_numpy(smodel.embed_utterance(ref)).unsqueeze(0).to(device)

    src, _ = librosa.load(args.source, sr=SR, mono=True)
    c = utils.get_content(cmodel, torch.from_numpy(src).unsqueeze(0).to(device))

    report = {"input_samples": len(src), "input_seconds": len(src) / SR, "outputs": {}}
    for name in args.presets.split(","):
        audio = convert(net_g, c, g, PRESETS[name], args.pitch, device)
        raw_len = len(audio)
        fitted = np.zeros(len(src), dtype=np.float32)
        fitted[:min(raw_len, len(src))] = audio[:len(src)]
        sf.write(outdir / f"{name}.wav", fitted, SR, subtype="PCM_16")
        report["outputs"][name] = {"raw_samples": raw_len, "raw_seconds": raw_len / SR,
                                   "diff_ms": 1000 * (raw_len - len(src)) / SR,
                                   "settings": dict(zip(("creak", "cpps", "h1h2", "h1a3", "pitch_var"),
                                                        PRESETS[name])) | {"pitch": args.pitch}}
        print(f"{name}: raw {raw_len} samples vs input {len(src)} "
              f"({report['outputs'][name]['diff_ms']:+.1f} ms)")
    (outdir / "convert_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
