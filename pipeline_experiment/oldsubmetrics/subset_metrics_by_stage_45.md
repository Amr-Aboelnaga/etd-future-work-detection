# Pipeline Experiment Results by Stage (Ground Truth Subset)

**Ground Truth Subset:** 129 files with 152 conclusion pages

Note: Only evaluates predictions for PDFs present in ground truth subset. Predictions for other PDFs are ignored.

## Stage 3 Only (Conclusion Detection)

| Model | Precision | Recall | F1 Score | Predictions | True Positives | False Positives | False Negatives |
|-------|-----------|--------|----------|-------------|----------------|-----------------|-----------------||
| llama4scout | 0.5115 | 0.7303 | 0.6016 | 217 | 111 | 106 | 41 |
| llama3.1-8b | 0.3065 | 0.6776 | 0.4221 | 336 | 103 | 233 | 49 |
| llama3.2-3b | 0.3529 | 0.1184 | 0.1773 | 51 | 18 | 33 | 134 |
| llama3.3 | 0.3587 | 0.7763 | 0.4906 | 329 | 118 | 211 | 34 |
| mistral-small | 0.5131 | 0.6447 | 0.5714 | 191 | 98 | 93 | 54 |

## Stage 1+3 (Layout + Conclusion Detection)

| Model | Precision | Recall | F1 Score | Predictions | True Positives | False Positives | False Negatives |
|-------|-----------|--------|----------|-------------|----------------|-----------------|-----------------||
| llama4scout | 0.5361 | 0.6842 | 0.6012 | 194 | 104 | 90 | 48 |
| llama3.1-8b | 0.3323 | 0.6842 | 0.4473 | 313 | 104 | 209 | 48 |
| llama3.2-3b | 0.3571 | 0.0987 | 0.1546 | 42 | 15 | 27 | 137 |
| llama3.3 | 0.3538 | 0.7961 | 0.4899 | 342 | 121 | 221 | 31 |
| mistral-small | 0.4688 | 0.6908 | 0.5585 | 224 | 105 | 119 | 47 |

## Stage 2+3 (Page Classification + Conclusion Detection)

| Model | Precision | Recall | F1 Score | Predictions | True Positives | False Positives | False Negatives |
|-------|-----------|--------|----------|-------------|----------------|-----------------|-----------------||
| llama4scout | 0.7203 | 0.6776 | 0.6983 | 143 | 103 | 40 | 49 |
| llama3.1-8b | 0.4466 | 0.6053 | 0.5140 | 206 | 92 | 114 | 60 |
| llama3.2-3b | 0.6667 | 0.1447 | 0.2378 | 33 | 22 | 11 | 130 |
| llama3.3 | 0.5135 | 0.7500 | 0.6096 | 222 | 114 | 108 | 38 |
| mistral-small | 0.7302 | 0.6053 | 0.6619 | 126 | 92 | 34 | 60 |

## Full Pipeline (Layout + Classification + Conclusion Detection)

| Model | Precision | Recall | F1 Score | Predictions | True Positives | False Positives | False Negatives |
|-------|-----------|--------|----------|-------------|----------------|-----------------|-----------------||
| llama4scout | 0.6667 | 0.6447 | 0.6555 | 147 | 98 | 49 | 54 |
| llama3.1-8b | 0.4811 | 0.6711 | 0.5604 | 212 | 102 | 110 | 50 |
| llama3.2-3b | 0.5588 | 0.1250 | 0.2043 | 34 | 19 | 15 | 133 |
| llama3.3 | 0.5197 | 0.7829 | 0.6247 | 229 | 119 | 110 | 33 |
| mistral-small | 0.7153 | 0.6776 | 0.6959 | 144 | 103 | 41 | 49 |

