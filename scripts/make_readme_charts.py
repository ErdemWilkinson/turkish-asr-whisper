"""Draw the README chart for the Common Voice CTC baseline into docs/ (light and dark SVG).

Character error rate, greedy decoding, 500 clips per test set, both models measured
with the same scripts/evaluate_asr.py on 2026-10-25.

    python scripts/make_readme_charts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from svgkit import grouped_bars, roadmap  # noqa: E402

DOCS = Path(__file__).resolve().parents[1] / "docs"


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    grouped_bars(DOCS / "asr-cer", "Character error rate of the CTC baseline (lower is better)",
                 "Greedy decoding, 500 clips per test set, same evaluation script. v11 is the published model.",
                 ["Common Voice Turkish, scripted", "ISSAI Turkish Speech Corpus, test"],
                 [("v9 · 2026-10-10", 2, [49.0, 50.1]), ("v11 · 2026-10-25", 0, [45.6, 46.3])])
    roadmap(DOCS / "roadmap-asr", "ASR roadmap",
            "Rule: the device model never grows; a wider model may only serve as a teacher. A step ships only if it passes its gate.",
            ["Next", "This week", "Later"],
            [("ASR", [[("Continue v11 on the GPU", "Gate: CER below 40% on both test sets", True)],
                      [("Wide teacher, 64k-parameter student", "Gate: the student beats v11 at the same size", False)],
                      [("Record the command dataset, train the command classifier", "Gate: the acceptance threshold in this README", False)]])])
    print("charts written to", DOCS)


if __name__ == "__main__":
    main()
