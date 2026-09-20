# Turkish Whisper-Command ASR

Training pipeline for a small, offline Turkish speech command recognizer,
targeting an [ESP32-P4-Pico](https://github.com/ErdemWilkinson/makeshift-flipper)
running fully disconnected. The goal is **not** general speech-to-text —
it's reliably classifying a short, fixed set of Turkish commands, including
when they're whispered at very low volume. A separate research-baseline
path (character-level CTC over Mozilla Common Voice) also lives here for
comparison, but is explicitly not the deployment target — see
[Two training paths](#two-training-paths) below.

This repository was split out of the makeshift-flipper firmware repo
because it's a separate concern (Python/TensorFlow training vs. ESP-IDF C
firmware) with its own, much larger and partly personal dataset.

**Status: the command-classifier pipeline and its data contract are
defined and scripted; recording the actual command dataset and clearing
the accuracy gate (see below) has not happened yet.** The Common Voice
baseline path has produced trained artifacts locally (not pushed — see
`.gitignore`) but is a research comparison, not a shipped model.

## Two training paths

| | `train.py` (command classifier) | `train_asr_common_voice.py` / `train_asr_smoke.py` (CTC baseline) |
|---|---|---|
| Task | Fixed-label classification (12 labels, see `commands.v1.json`) | Open-vocabulary character-level transcription |
| Deployment target | **Yes** — this is what ships to the P4 | **No** — research baseline only, both scripts say so in their own docstrings |
| Input | 1-second 16kHz clips of the 12 known commands/wake word/unknown/silence | Common Voice or paired Wispr History audio+transcript |
| Split | Speaker-disjoint (same voice never in both train and validation) | Mozilla's official train/dev split, preserved as-is |
| Why it exists | The actual product feature | A sanity check on how far a from-scratch Turkish CTC model gets with public/personal data, useful context before trusting the classifier's numbers |

If you only care about the shipped feature, you want `prepare_dataset.py` →
`train.py` → `export_tflite.py`. The rest of this README covers both paths
since they share data-handling conventions.

## Command set

Defined in [`commands.v1.json`](commands.v1.json) — keep the label order
fixed once a model is on a device, since the trained model's output index
order follows it:

```
unknown, silence, wake, menu_open, back, wifi_scan, wifi_monitor,
bluetooth_scan, rfid_read, ir_send, errors_show, cancel
```

`wake` = "hey flipper"; the rest map to specific makeshift-flipper menu
actions (see the `phrases` field in the JSON for the exact Turkish phrase
list per label). `unknown` and `silence` are mandatory negative classes —
without them the model can't tell "not a command" from "a command I don't
recognize."

## Repository layout

```text
voice/
├── scripts/
│   ├── prepare_dataset.py            # validates voice/data/raw/, writes a manifest
│   ├── train.py                       # trains the command classifier (deployment target)
│   ├── export_tflite.py               # float -> full-int8 .tflite
│   ├── train_asr_smoke.py             # small CTC baseline smoke test
│   ├── train_asr_common_voice.py      # full CTC baseline on Common Voice
│   ├── import_common_voice_scripted.py       # Common Voice Scripted Speech -> manifest
│   ├── import_common_voice_spontaneous.py    # Common Voice Spontaneous Speech -> manifest
│   ├── build_quality_common_voice.py  # metadata-validated Common Voice subset
│   ├── export_wispr_pairs.py          # exports paired Wispr Flow History rows (see below)
│   ├── inspect_wispr_schema.py        # prints only schema/row counts, never transcripts
│   ├── inspect_wispr_history_stats.py # prints non-sensitive availability stats
│   ├── inspect_wispr_audio_format.py  # identifies audio container metadata
│   └── finalize_common_voice.ps1
├── commands.v1.json                   # tracked: the versioned label/phrase set
├── QUALITY_RECORDING_PROTOCOL.md      # tracked: the v2 controlled-recording protocol
├── data/                              # NOT tracked: all raw/imported/derived audio
└── artifacts/                         # NOT tracked: trained models
```

**What's tracked in git and what isn't:** only code and small config/docs
(`commands.v1.json`, the recording protocol) are tracked. All audio —
raw recordings, imported Common Voice clips, exported Wispr pairs, and
every trained model artifact — is gitignored. This is not just a size
concern: some of this data is personal voice recordings and must never be
published. Rebuild what you need locally with the commands below.

## Data contract (command classifier)

```text
voice/data/raw/<speaker>/<style>/<label>/<uuid>.wav
```

Example: `voice/data/raw/erdem/whisper/wifi_scan/001.wav`

- WAV, mono, 16-bit PCM, 16kHz.
- One command per recording, ~150ms of silence padding at start/end.
- At least 80 recordings per label per style, at least 8 distinct speakers.
- `whisper`-style recordings: natural whisper, 15–40cm from the mic.
- `unknown`: Turkish speech the model should *not* match to any command.
  `silence`: ambient room noise. Both are required to keep false-trigger
  rate down.

For a more controlled, higher-quality v2 protocol (same mic across
speakers, scripted sentences, explicit accept/reject criteria), see
[QUALITY_RECORDING_PROTOCOL.md](QUALITY_RECORDING_PROTOCOL.md) — target:
≥12 speakers × 80 sentences × 2 reads (~1,920 recordings), 8/2/2 speaker
split across train/dev/test so no speaker's voice crosses a split boundary.

## Training pipeline (command classifier — the deployment target)

From this directory, Python 3.11:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-train.txt

python scripts\prepare_dataset.py     # validates data/raw/, writes a manifest (no audio conversion)
python scripts\train.py               # speaker-disjoint split, writes artifacts/
python scripts\export_tflite.py       # float -> full-int8 .tflite
```

`train.py` performs a speaker-based split explicitly so the same person's
voice can never appear in both the training and evaluation sets — without
that, accuracy numbers would be inflated by the model partially recognizing
a specific voice rather than the command. Whisper-subset accuracy is
reported separately from normal-volume accuracy.

## Common Voice baseline path (research comparison, not deployed)

```powershell
# 1. Download & extract "Common Voice Spontaneous Speech -- Turkish" from
#    the Mozilla Data Collective, then build a manifest from its TSV:
python scripts\import_common_voice_spontaneous.py <extracted-archive-dir>

# Or, for the Scripted Speech variant (keeps Mozilla's official train/dev/test split):
python scripts\import_common_voice_scripted.py <extracted-archive-dir>

# 2. Optional: a conservative, metadata-filtered subset (bad text, implausible
#    duration/rate, duplicates, speaker overrepresentation -- not a noise judgment):
python scripts\build_quality_common_voice.py

# 3. Train the CTC baseline:
python scripts\train_asr_smoke.py             # small/fast sanity check
python scripts\train_asr_common_voice.py      # full run
python scripts\export_tflite.py
```

Keep the raw extracted archive under `voice/data/source/` (gitignored, not
committed). The importer scripts only accept audio/text pairs resolvable by
an explicit filename column in the archive's metadata — they never guess a
pairing by timestamp or ordering.

### Wispr Flow History (personal data, opt-in, local-only)

`export_wispr_pairs.py` can pull paired WAV/transcript rows out of a local
[Wispr Flow](https://wisprflow.ai/) History SQLite database (opened
strictly read-only) as an additional personal-voice data source:

```powershell
python scripts\inspect_wispr_schema.py <path-to-History.db>          # schema + row counts only
python scripts\inspect_wispr_history_stats.py <path-to-History.db>   # non-sensitive availability stats
python scripts\export_wispr_pairs.py <path-to-History.db> --output data\wispr
```

The `inspect_*` scripts deliberately print only schema/counts/format
metadata — never transcript content — so you can sanity-check a database
without a second script accidentally leaking its contents to a terminal
that might get pasted somewhere. `export_wispr_pairs.py` only emits rows
where the WAV blob and transcript come from the *same* History row (never
inferred by matching nearby timestamps). None of this is your data to run
against someone else's device — it's designed for exporting your own
local history.

## Acceptance threshold (command classifier)

Before integration into the P4 firmware, on a held-out set recorded by
speakers not in training:

- Whisper command accuracy: ≥ 85%
- Normal-volume command accuracy: ≥ 92%
- `unknown`/`silence` false-accept rate: ≤ 2%

A model that doesn't clear this bar does not get added to firmware. Open
Turkish general-speech corpora (Common Voice) can help pretraining/
background diversity, but do not substitute for actual whispered recordings
of the target commands — that's why the two paths in this repo stay separate.

## Related repositories

- [makeshift-flipper](https://github.com/ErdemWilkinson/makeshift-flipper) —
  the ESP32-P4/C6 firmware this model is meant to run on
- [turkish-ocr-tinyml](https://github.com/ErdemWilkinson/turkish-ocr-tinyml) —
  the sibling TinyML pipeline (offline Turkish line OCR), split out for the
  same reason (separate concern, separate dataset)
