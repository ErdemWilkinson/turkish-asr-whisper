# Quality Turkish ASR recording protocol

Common Voice provides diversity; real on-device performance also needs
supervised recordings collected with the same microphone. Target for v1: at
least 12 speakers, each reading 80 different short sentences twice. That's
roughly 1,920 recordings.

Each recording should be mono, 16 kHz, 16-bit PCM WAV. The speaker sits
15–30 cm from the microphone and speaks at a natural pace. Leave 200 ms of
silence before and after each sentence. The second take of the same
sentence is recorded in a different room, or with low-level background
noise.

Recordings are kept in this layout; this folder is git-ignored:

```text
voice/data/raw_v2/<speaker_id>/<session>/<prompt_id>.wav
voice/data/raw_v2/<speaker_id>/<session>/transcript.tsv
```

`transcript.tsv` header: `prompt_id<TAB>transcript`. Keep separate test
speakers for the device: 8 people for training, 2 for validation, 2 for
test. The same person's recordings must never end up in more than one
split.

Rejection criteria: cut-off words, noticeable clipping, wrong/missing
transcript, overlapping speech, recordings shorter than 0.8s or longer than
12s. If the whisper-volume target is reintroduced, add an extra `whisper`
session per speaker to the same protocol.
