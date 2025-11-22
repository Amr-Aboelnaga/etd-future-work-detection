# Pipeline Experiment Results by Stage (Ground Truth Subset)

**Ground Truth Subset:** 41 files with 41 conclusion pages

Note: Only evaluates predictions for PDFs present in ground truth subset. Predictions for other PDFs are ignored.

## Stage 3 Only (Conclusion Detection)

| Model | Precision | Recall | F1 Score | Predictions | True Positives | False Positives | False Negatives |
|-------|-----------|--------|----------|-------------|----------------|-----------------|-----------------||
| llama4scout | 0.5345 | 0.7561 | 0.6263 | 58 | 31 | 27 | 10 |
| llama3.1-8b | 0.2981 | 0.7561 | 0.4276 | 104 | 31 | 73 | 10 |
| llama3.2-3b | 0.3077 | 0.0976 | 0.1481 | 13 | 4 | 9 | 37 |
| llama3.3 | 0.3763 | 0.8537 | 0.5224 | 93 | 35 | 58 | 6 |
| mistral-small | 0.5254 | 0.7561 | 0.6200 | 59 | 31 | 28 | 10 |

## Stage 1+3 (Layout + Conclusion Detection)

| Model | Precision | Recall | F1 Score | Predictions | True Positives | False Positives | False Negatives |
|-------|-----------|--------|----------|-------------|----------------|-----------------|-----------------||
| llama4scout | 0.5577 | 0.7073 | 0.6237 | 52 | 29 | 23 | 12 |
| llama3.1-8b | 0.3465 | 0.8537 | 0.4930 | 101 | 35 | 66 | 6 |
| llama3.2-3b | 0.3750 | 0.1463 | 0.2105 | 16 | 6 | 10 | 35 |
| llama3.3 | 0.3737 | 0.9024 | 0.5286 | 99 | 37 | 62 | 4 |
| mistral-small | 0.4783 | 0.8049 | 0.6000 | 69 | 33 | 36 | 8 |

## Stage 2+3 (Page Classification + Conclusion Detection)

| Model | Precision | Recall | F1 Score | Predictions | True Positives | False Positives | False Negatives |
|-------|-----------|--------|----------|-------------|----------------|-----------------|-----------------||
| llama4scout | 0.7576 | 0.6098 | 0.6757 | 33 | 25 | 8 | 16 |
| llama3.1-8b | 0.5000 | 0.6341 | 0.5591 | 52 | 26 | 26 | 15 |
| llama3.2-3b | 0.8750 | 0.1707 | 0.2857 | 8 | 7 | 1 | 34 |
| llama3.3 | 0.5926 | 0.7805 | 0.6737 | 54 | 32 | 22 | 9 |
| mistral-small | 0.8333 | 0.6098 | 0.7042 | 30 | 25 | 5 | 16 |

## Full Pipeline (Layout + Classification + Conclusion Detection)

| Model | Precision | Recall | F1 Score | Predictions | True Positives | False Positives | False Negatives |
|-------|-----------|--------|----------|-------------|----------------|-----------------|-----------------||
| llama4scout | 0.8065 | 0.6098 | 0.6944 | 31 | 25 | 6 | 16 |
| llama3.1-8b | 0.5741 | 0.7561 | 0.6526 | 54 | 31 | 23 | 10 |
| llama3.2-3b | 0.8571 | 0.1463 | 0.2500 | 7 | 6 | 1 | 35 |
| llama3.3 | 0.6250 | 0.8537 | 0.7216 | 56 | 35 | 21 | 6 |
| mistral-small | 0.8158 | 0.7561 | 0.7848 | 38 | 31 | 7 | 10 |

