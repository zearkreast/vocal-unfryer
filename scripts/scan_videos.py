"""Find the 30 s window with the most vocal fry in each downloaded video."""
import sys
from pathlib import Path

import numpy as np

from fry_metrics import CREAK_THRESHOLD, HOP_S, ROOT, creak_track, to_wav

WIN_S, STEP_S, MIN_VOICED = 30.0, 5.0, 0.35

out_dir = ROOT / "work" / "scan"
out_dir.mkdir(parents=True, exist_ok=True)

for src in sorted(Path(sys.argv[1]).glob("*.m4a")):
    wav = to_wav(src, out_dir / f"{src.stem}_16k.wav")
    y, voiced = creak_track(wav)
    creaky = (y >= CREAK_THRESHOLD) & voiced
    win, step = int(WIN_S / HOP_S), int(STEP_S / HOP_S)
    rows = []
    for i in range(0, max(1, len(y) - win + 1), step):
        v = voiced[i:i + win].sum()
        if v / win < MIN_VOICED:
            continue
        rows.append((100.0 * creaky[i:i + win].sum() / v, i * HOP_S, 100.0 * v / win))
    rows.sort(reverse=True)
    total = 100.0 * creaky.sum() / max(1, voiced.sum())
    print(f"\n{src.stem}: {len(y) * HOP_S / 60:.1f} min, creak {total:.1f}% of voiced frames overall")
    for pct, start, vpct in rows[:3]:
        print(f"  {int(start // 60)}:{start % 60:04.1f}  creak {pct:5.1f}%  (voiced {vpct:.0f}% of window)")
    least = min(rows) if rows else None
    if least:
        print(f"  least creaky window: {int(least[1] // 60)}:{least[1] % 60:04.1f}  creak {least[0]:.1f}%")
