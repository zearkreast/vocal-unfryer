# Vocal Fry Remover: Project Brief

## Goal
Build a tool that takes an audio or video file (including a YouTube video downloaded for personal use) and outputs the same speech with vocal fry (creaky voice) reduced, so it can be listened to directly. No transcription and re-voicing.

## Background
- No dedicated, reliable "vocal fry remover" was found. Noise reduction and EQ (Audacity, Audition, iZotope RX) do not fix it, because fry is a voicing mode, not added noise.
- Closest existing options (untested on fry): ElevenLabs voice changer, RVC (open-source voice conversion), Adobe Podcast Enhance Speech.
- Caveat: many voice conversion models preserve prosody and pitch, so fry may carry through unless the pitch contour is edited.

## Constraints
- Output duration must match the input exactly, so the video stays in sync.
- Keep the speaker's own timbre; avoid a robotic or "AI voice" sound.
- Offer a strength control. Target "less fry, still natural", not total removal.
- Personal use only. Downloading from YouTube may conflict with its terms.

## Phases

### Phase 0: Detect creak (1-2 weeks)
- [ ] Collect 10-20 clips with heavy fry plus a few clean controls
- [ ] Extract pitch and voicing with Praat/Parselmouth
- [ ] Flag creaky segments: very low and irregular pitch, irregular pulses, low harmonicity
- [ ] Report the percentage of each clip that is creaky

### Phase 1: Signal-processing fix (2-3 weeks)
- [ ] In flagged regions, raise and smooth the pitch contour
- [ ] Resynthesize with PSOLA or WORLD
- [ ] Verify duration is unchanged
- [ ] Note artifacts (irregular glottal pulses are hard for PSOLA)

### Phase 2: Neural resynthesis (4-8 weeks)
- [ ] Extract content features (HuBERT or Whisper encoder), speaker embedding, and pitch contour
- [ ] Replace creaky parts of the pitch contour with a smooth natural one
- [ ] Resynthesize with a neural vocoder (HiFi-GAN style), keeping the speaker's timbre

### Phase 3: Dedicated fry-to-clear model (optional, 2-3 months)
- [ ] Build synthetic training pairs by adding fry to clean speech (lowered pitch, irregular pulses, roughness)
- [ ] Train a model to reverse it
- [ ] Fine-tune on real fry recordings

### Phase 4: Packaging (2-3 weeks)
- [ ] CLI: URL or file -> ffmpeg audio extraction -> processing -> remux with original video
- [ ] Simple web or desktop UI with strength slider and A/B preview
- [ ] Stretch: real-time processing for browser playback

### Phase 5: Evaluation (ongoing)
- [ ] Creak rate before and after (Phase 0 detector)
- [ ] Intelligibility: Whisper word error rate before and after
- [ ] Listening tests for naturalness and speaker similarity

## Suggested first steps for Claude Code
1. Set up a Python project with parselmouth, librosa, pyworld, ffmpeg-python, and pytest.
2. Implement `detect_creak(audio_path) -> list[(start, end)]` and a script that prints the creaky percentage per clip.
3. Implement the Phase 1 pitch-smoothing resynthesis behind a `--strength` flag.
4. Add a CLI that extracts audio from a video, processes it, and remuxes it with the original video stream.
5. Add an evaluation script that runs Whisper WER and the creak detector on before/after files.
