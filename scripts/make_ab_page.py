"""Build a self-contained A/B listening page (work/ab.html) from one or more evaluated output folders."""
import base64
import html
import io
import json
import sys
from pathlib import Path

import soundfile as sf

clips = Path(sys.argv[1]).resolve()
outdirs = [Path(p).resolve() for p in sys.argv[2:]]
page = clips.parent / "ab.html"

LABELS = {
    "neutral": "average voice (creak 0)",
    "less_fry": "less fry (creak -1)",
    "much_less_fry": "much less fry (creak -2)",
    "more_fry": "more fry (creak +3, sanity check)",
}


def embed(p):
    """Lossless FLAC data URI, so the page works as a single file."""
    audio, sr = sf.read(p)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="FLAC")
    return "data:audio/flac;base64," + base64.b64encode(buf.getvalue()).decode()


def stats(m):
    return (f"<dl><div><dt>Creaky frames</dt><dd>{m['creak_pct']}%</dd></div>"
            f"<div><dt>Median pitch</dt><dd>{m['median_f0_hz']} Hz</dd></div>"
            f"<div><dt>Voice match</dt><dd>{m['speaker_sim_to_reference']:.2f}</dd></div>"
            f"<div><dt>Word differences</dt><dd>{m['wer_vs_original_pct']}%</dd></div>"
            f"<div><dt>Quality estimate</dt><dd>{m['pesq_estimate']}</dd></div>"
            f"<div><dt>Sync start / end</dt><dd>{m['lag_first_5s_ms']:+.0f} / {m['lag_last_5s_ms']:+.0f} ms</dd></div></dl>")


def card(label, src, m=None):
    return (f"<section class=card><h2>{html.escape(label)}</h2>"
            f"<audio controls preload=metadata src=\"{src}\"></audio>{stats(m) if m else ''}</section>")


first = json.loads((outdirs[0] / "eval_report.json").read_text())
calib = first["calibration"]
cards = [card("Original (full quality)", embed(clips / "test_fry_48k.wav")),
         card("Original (16 kHz, what the model hears)", embed(clips / "test_fry_16k.wav"), first["rows"]["original"])]
for outdir in outdirs:
    rows = json.loads((outdir / "eval_report.json").read_text())["rows"]
    conv = json.loads((outdir / "convert_report.json").read_text())["outputs"]
    for name, m in rows.items():
        if name == "original":
            continue
        pitch = conv[name]["settings"]["pitch"]
        cards.append(card(f"VQVC, pitch level {pitch:+g}: {LABELS.get(name, name)}", embed(outdir / f"{name}.wav"), m))

page.write_text(f"""<!doctype html>
<html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Vocal Fry A/B</title>
<style>
:root {{ --bg:#f7f7f5; --card:#fff; --text:#1d1d1b; --muted:#6b6b66; --line:#e3e3de; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#161615; --card:#1f1f1d; --text:#ecece8; --muted:#a3a39c; --line:#33332f; }} }}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.5 system-ui,sans-serif; }}
main {{ max-width:760px; margin:0 auto; padding:32px 16px 64px; }}
h1 {{ font-size:22px; margin:0 0 4px }} p {{ color:var(--muted); margin:0 0 20px }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; margin-bottom:12px }}
h2 {{ font-size:15px; margin:0 0 10px }} audio {{ width:100% }}
dl {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:8px 16px; margin:12px 0 0 }}
dt {{ color:var(--muted); font-size:12px }} dd {{ margin:0; font-variant-numeric:tabular-nums }}
.note {{ font-size:13px; color:var(--muted); margin-top:20px }}
</style></head><body><main>
<h1>Vocal fry A/B test</h1>
<p>Same 30 seconds of speech, run through VoiceQualityVC with different creak and pitch-level settings. All versions are exactly the same length.</p>
{''.join(cards)}
<p class=note>Creaky frames: CreaPy detections as a share of voiced frames. Voice match: ECAPA cosine similarity to a clean clip of the same speaker; for scale, the original scores {calib['same_speaker_original_vs_reference']:.2f} and a different speaker {calib['different_speaker_vs_reference']:.2f}.
Word differences: Whisper transcript of each version vs the original's. Quality estimate: torchaudio SQUIM's reference-free PESQ guess (1 to 4.5); it counts fry as distortion, so it is not a naturalness score.</p>
</main></body></html>
""", encoding="utf-8")
print(page)
