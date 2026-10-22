"""Per video: find the main speaker, then that speaker's most and least creaky 30 s windows."""
import sys
from pathlib import Path

import numpy as np
import torch

from evaluate import SR, Speaker, load
from fry_metrics import CREAK_THRESHOLD, HOP_S, creak_track

CHUNK_S, WIN_CHUNKS, SAME_SPK, MIN_VOICED = 5, 6, 0.35, 0.30
FR = int(CHUNK_S / HOP_S)

spk = Speaker()
for wav_path in sys.argv[1:]:
    wav_path = Path(wav_path)
    cache = wav_path.with_suffix(".creak.npz")
    if cache.exists():
        d = np.load(cache)
        y, voiced = d["y"], d["voiced"]
    else:
        y, voiced = creak_track(wav_path)
        np.savez(cache, y=y, voiced=voiced)
    wav = load(wav_path)
    n = min(len(wav) // (SR * CHUNK_S), len(y) // FR)

    vfrac = np.array([voiced[i * FR:(i + 1) * FR].mean() for i in range(n)])
    creaky = (y >= CREAK_THRESHOLD) & voiced
    cfrac = np.array([creaky[i * FR:(i + 1) * FR].sum() / max(1, voiced[i * FR:(i + 1) * FR].sum())
                      for i in range(n)])
    emb = torch.stack([spk.embed(wav[i * SR * CHUNK_S:(i + 1) * SR * CHUNK_S]) for i in range(n)])
    emb = torch.nn.functional.normalize(emb, dim=1)
    sim = (emb @ emb.T).numpy()

    speech = vfrac >= MIN_VOICED
    medoid = np.argmax(np.where(speech, (sim * speech).sum(1), -np.inf))
    main = speech & (sim[medoid] >= SAME_SPK)
    print(f"\n{wav_path.stem}: {main.sum()}/{speech.sum()} speech chunks match the main speaker "
          f"(medoid at {medoid * CHUNK_S // 60}:{medoid * CHUNK_S % 60:02d})")

    wins = []
    for s in range(n - WIN_CHUNKS + 1):
        idx = slice(s, s + WIN_CHUNKS)
        if main[idx].all():
            v = voiced[s * FR:(s + WIN_CHUNKS) * FR]
            c = creaky[s * FR:(s + WIN_CHUNKS) * FR]
            wins.append((100 * c.sum() / v.sum(), s * CHUNK_S, 100 * v.mean()))
    if not wins:
        print("  no 30 s window with only the main speaker")
        continue
    wins.sort(reverse=True)
    for pct, t, vp in wins[:3]:
        print(f"  most creaky: {t // 60}:{t % 60:02d}  creak {pct:.1f}%  voiced {vp:.0f}%")
    pct, t, vp = wins[-1]
    print(f"  least creaky: {t // 60}:{t % 60:02d}  creak {pct:.1f}%  voiced {vp:.0f}%")
