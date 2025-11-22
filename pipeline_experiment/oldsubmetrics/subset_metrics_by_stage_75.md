# Pipeline Experiment Results by Stage (Ground Truth Subset)

**Ground Truth Subset:** 222 files with 254 conclusion pages

Note: Only evaluates predictions for PDFs present in ground truth subset. Predictions for other PDFs are ignored.

## Stage 3 Only (Conclusion Detection)

| Model | Precision | Recall | F1 Score | Predictions | True Positives | False Positives | False Negatives |
|-------|-----------|--------|----------|-------------|----------------|-----------------|-----------------||
| llama4scout | 0.5053 | 0.7520 | 0.6044 | 378 | 191 | 187 | 63 |
| llama3.1-8b | 0.2984 | 0.7165 | 0.4213 | 610 | 182 | 428 | 72 |
| llama3.2-3b | 0.4000 | 0.1732 | 0.2418 | 110 | 44 | 66 | 210 |
| llama3.3 | 0.3616 | 0.8071 | 0.4994 | 567 | 205 | 362 | 49 |
| mistral-small | 0.4955 | 0.6535 | 0.5637 | 335 | 166 | 169 | 88 |

## Stage 1+3 (Layout + Conclusion Detection)

| Model | Precision | Recall | F1 Score | Predictions | True Positives | False Positives | False Negatives |
|-------|-----------|--------|----------|-------------|----------------|-----------------|-----------------||
| llama4scout | 0.5262 | 0.7126 | 0.6054 | 344 | 181 | 163 | 73 |
| llama3.1-8b | 0.3143 | 0.6929 | 0.4324 | 560 | 176 | 384 | 78 |
| llama3.2-3b | 0.4268 | 0.1378 | 0.2083 | 82 | 35 | 47 | 219 |
| llama3.3 | 0.3497 | 0.7874 | 0.4843 | 572 | 200 | 372 | 54 |
| mistral-small | 0.4767 | 0.6850 | 0.5622 | 365 | 174 | 191 | 80 |

## Stage 2+3 (Page Classification + Conclusion Detection)

| Model | Precision | Recall | F1 Score | Predictions | True Positives | False Positives | False Negatives |
|-------|-----------|--------|----------|-------------|----------------|-----------------|-----------------||
| llama4scout | 0.7027 | 0.7165 | 0.7096 | 259 | 182 | 77 | 72 |
| llama3.1-8b | 0.4397 | 0.6457 | 0.5231 | 373 | 164 | 209 | 90 |
| llama3.2-3b | 0.6875 | 0.1732 | 0.2767 | 64 | 44 | 20 | 210 |
| llama3.3 | 0.5128 | 0.7913 | 0.6223 | 392 | 201 | 191 | 53 |
| mistral-small | 0.7067 | 0.6260 | 0.6639 | 225 | 159 | 66 | 95 |

## Full Pipeline (Layout + Classification + Conclusion Detection)

| Model | Precision | Recall | F1 Score | Predictions | True Positives | False Positives | False Negatives |
|-------|-----------|--------|----------|-------------|----------------|-----------------|-----------------||
| llama4scout | 0.6615 | 0.6772 | 0.6693 | 260 | 172 | 88 | 82 |
| llama3.1-8b | 0.4626 | 0.6811 | 0.5510 | 374 | 173 | 201 | 81 |
| llama3.2-3b | 0.6111 | 0.1299 | 0.2143 | 54 | 33 | 21 | 221 |
| llama3.3 | 0.5196 | 0.7835 | 0.6248 | 383 | 199 | 184 | 55 |
| mistral-small | 0.7020 | 0.6772 | 0.6894 | 245 | 172 | 73 | 82 |

