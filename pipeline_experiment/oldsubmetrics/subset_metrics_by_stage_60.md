# Pipeline Experiment Results by Stage (Ground Truth Subset)

**Ground Truth Subset:** 176 files with 203 conclusion pages

Note: Only evaluates predictions for PDFs present in ground truth subset. Predictions for other PDFs are ignored.

## Stage 3 Only (Conclusion Detection)

| Model | Precision | Recall | F1 Score | Predictions | True Positives | False Positives | False Negatives |
|-------|-----------|--------|----------|-------------|----------------|-----------------|-----------------||
| llama4scout | 0.4886 | 0.7389 | 0.5882 | 307 | 150 | 157 | 53 |
| llama3.1-8b | 0.2904 | 0.6995 | 0.4104 | 489 | 142 | 347 | 61 |
| llama3.2-3b | 0.3718 | 0.1429 | 0.2064 | 78 | 29 | 49 | 174 |
| llama3.3 | 0.3457 | 0.7833 | 0.4796 | 460 | 159 | 301 | 44 |
| mistral-small | 0.4847 | 0.6256 | 0.5462 | 262 | 127 | 135 | 76 |

## Stage 1+3 (Layout + Conclusion Detection)

| Model | Precision | Recall | F1 Score | Predictions | True Positives | False Positives | False Negatives |
|-------|-----------|--------|----------|-------------|----------------|-----------------|-----------------||
| llama4scout | 0.5222 | 0.6946 | 0.5962 | 270 | 141 | 129 | 62 |
| llama3.1-8b | 0.3026 | 0.6798 | 0.4188 | 456 | 138 | 318 | 65 |
| llama3.2-3b | 0.3651 | 0.1133 | 0.1729 | 63 | 23 | 40 | 180 |
| llama3.3 | 0.3412 | 0.7833 | 0.4753 | 466 | 159 | 307 | 44 |
| mistral-small | 0.4674 | 0.6700 | 0.5506 | 291 | 136 | 155 | 67 |

## Stage 2+3 (Page Classification + Conclusion Detection)

| Model | Precision | Recall | F1 Score | Predictions | True Positives | False Positives | False Negatives |
|-------|-----------|--------|----------|-------------|----------------|-----------------|-----------------||
| llama4scout | 0.6946 | 0.6946 | 0.6946 | 203 | 141 | 62 | 62 |
| llama3.1-8b | 0.4051 | 0.6207 | 0.4903 | 311 | 126 | 185 | 77 |
| llama3.2-3b | 0.6531 | 0.1576 | 0.2540 | 49 | 32 | 17 | 171 |
| llama3.3 | 0.4859 | 0.7635 | 0.5939 | 319 | 155 | 164 | 48 |
| mistral-small | 0.6954 | 0.5961 | 0.6419 | 174 | 121 | 53 | 82 |

## Full Pipeline (Layout + Classification + Conclusion Detection)

| Model | Precision | Recall | F1 Score | Predictions | True Positives | False Positives | False Negatives |
|-------|-----------|--------|----------|-------------|----------------|-----------------|-----------------||
| llama4scout | 0.6617 | 0.6552 | 0.6584 | 201 | 133 | 68 | 70 |
| llama3.1-8b | 0.4416 | 0.6700 | 0.5323 | 308 | 136 | 172 | 67 |
| llama3.2-3b | 0.5250 | 0.1034 | 0.1728 | 40 | 21 | 19 | 182 |
| llama3.3 | 0.5016 | 0.7783 | 0.6100 | 315 | 158 | 157 | 45 |
| mistral-small | 0.6891 | 0.6552 | 0.6717 | 193 | 133 | 60 | 70 |

