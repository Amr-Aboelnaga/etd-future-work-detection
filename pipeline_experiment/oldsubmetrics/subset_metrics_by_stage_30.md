# Pipeline Experiment Results by Stage (Ground Truth Subset)

**Ground Truth Subset:** 88 files with 101 conclusion pages

Note: Only evaluates predictions for PDFs present in ground truth subset. Predictions for other PDFs are ignored.

## Stage 3 Only (Conclusion Detection)

| Model | Precision | Recall | F1 Score | Predictions | True Positives | False Positives | False Negatives |
|-------|-----------|--------|----------|-------------|----------------|-----------------|-----------------||
| llama4scout | 0.5588 | 0.7525 | 0.6414 | 136 | 76 | 60 | 25 |
| llama3.1-8b | 0.3241 | 0.6931 | 0.4416 | 216 | 70 | 146 | 31 |
| llama3.2-3b | 0.3226 | 0.0990 | 0.1515 | 31 | 10 | 21 | 91 |
| llama3.3 | 0.3832 | 0.8119 | 0.5206 | 214 | 82 | 132 | 19 |
| mistral-small | 0.5820 | 0.7030 | 0.6368 | 122 | 71 | 51 | 30 |

## Stage 1+3 (Layout + Conclusion Detection)

| Model | Precision | Recall | F1 Score | Predictions | True Positives | False Positives | False Negatives |
|-------|-----------|--------|----------|-------------|----------------|-----------------|-----------------||
| llama4scout | 0.5760 | 0.7129 | 0.6372 | 125 | 72 | 53 | 29 |
| llama3.1-8b | 0.3578 | 0.7228 | 0.4787 | 204 | 73 | 131 | 28 |
| llama3.2-3b | 0.4074 | 0.1089 | 0.1719 | 27 | 11 | 16 | 90 |
| llama3.3 | 0.3784 | 0.8317 | 0.5201 | 222 | 84 | 138 | 17 |
| mistral-small | 0.5000 | 0.7327 | 0.5944 | 148 | 74 | 74 | 27 |

## Stage 2+3 (Page Classification + Conclusion Detection)

| Model | Precision | Recall | F1 Score | Predictions | True Positives | False Positives | False Negatives |
|-------|-----------|--------|----------|-------------|----------------|-----------------|-----------------||
| llama4scout | 0.7816 | 0.6733 | 0.7234 | 87 | 68 | 19 | 33 |
| llama3.1-8b | 0.5083 | 0.6040 | 0.5520 | 120 | 61 | 59 | 40 |
| llama3.2-3b | 0.7059 | 0.1188 | 0.2034 | 17 | 12 | 5 | 89 |
| llama3.3 | 0.5985 | 0.7822 | 0.6781 | 132 | 79 | 53 | 22 |
| mistral-small | 0.8421 | 0.6337 | 0.7232 | 76 | 64 | 12 | 37 |

## Full Pipeline (Layout + Classification + Conclusion Detection)

| Model | Precision | Recall | F1 Score | Predictions | True Positives | False Positives | False Negatives |
|-------|-----------|--------|----------|-------------|----------------|-----------------|-----------------||
| llama4scout | 0.7528 | 0.6634 | 0.7053 | 89 | 67 | 22 | 34 |
| llama3.1-8b | 0.5349 | 0.6832 | 0.6000 | 129 | 69 | 60 | 32 |
| llama3.2-3b | 0.5714 | 0.1188 | 0.1967 | 21 | 12 | 9 | 89 |
| llama3.3 | 0.6074 | 0.8119 | 0.6949 | 135 | 82 | 53 | 19 |
| mistral-small | 0.8161 | 0.7030 | 0.7553 | 87 | 71 | 16 | 30 |

