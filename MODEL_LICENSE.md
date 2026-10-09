# License for trained models

The [MIT License](LICENSE) in this repository covers the source code only.
It does not cover trained model files (`*.keras`, `*.tflite`) produced by
that code, whether or not they are distributed with this repository.

A trained model carries the terms of the data it was trained on:

| Training data | Terms for the resulting model |
|---|---|
| Mozilla Common Voice (CC0), your own recordings | No third-party restriction |
| Multilingual Spoken Words Corpus (CC BY 4.0) | Attribution to MLCommons MSWC required |
| ISSAI Turkish Speech Corpus (MIT) | Keep the ISSAI copyright and license notice |
| Turkish Speech Command Dataset (CC BY-NC-SA 4.0) | **Non-commercial use only, share-alike, attribution required** |

Any model trained with the Turkish Speech Command Dataset is released under
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

As of 2026-10-09 no model in this project has been trained with that
dataset; the CTC baseline uses Common Voice only.

## Attribution

- Mozilla Common Voice, Turkish. <https://datacollective.mozillafoundation.org/>
- Multilingual Spoken Words Corpus, MLCommons, CC BY 4.0.
  <https://huggingface.co/datasets/MLCommons/ml_spoken_words>
- Turkish Speech Command Dataset, Murat Kurtkaya (2021), CC BY-NC-SA 4.0.
  <https://www.kaggle.com/datasets/muratkurtkaya/turkish-speech-command-dataset>
- ISSAI Turkish Speech Corpus, MIT.
  <https://huggingface.co/datasets/issai/Turkish_Speech_Corpus>
