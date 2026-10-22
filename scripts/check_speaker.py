"""Check that candidate windows hold one speaker: 5 s chunks vs a reference window (ECAPA cosine)."""
import sys

import numpy as np

from evaluate import SR, Speaker, load, median_f0

ref = load(sys.argv[1])
spk = Speaker()
ref_emb = spk.embed(ref)
for path in sys.argv[2:]:
    wav = load(path)
    sims = []
    for i in range(0, len(wav) - SR * 5 + 1, SR * 5):
        import torch
        sims.append(float(torch.nn.functional.cosine_similarity(spk.embed(wav[i:i + SR * 5]), ref_emb, dim=0)))
    print(f"{path}: median F0 {median_f0(wav, 40, 500):.0f} Hz, "
          f"5 s chunk similarity to reference: {' '.join(f'{s:.2f}' for s in sims)}")
print(f"reference median F0 {median_f0(ref, 40, 500):.0f} Hz")
