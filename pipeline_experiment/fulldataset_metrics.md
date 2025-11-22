# Pipeline Experiment Results by Stage

## Stage 3 Only (Conclusion Detection)

| Model | Precision | Recall | F1 Score | Predictions | True Positives | False Positives | False Negatives | Duration (s) |
|-------|-----------|--------|----------|-------------|----------------|-----------------|-----------------|--------------|
| LLaMA 4 Scout | 0.5399 | 0.7788 | 0.6377 | 489 | 264 | 225 | 75 | 0.0 |
| LLaMA 3.1 8B | 0.3028 | 0.7404 | 0.4298 | 829 | 251 | 578 | 88 | 966.41 |
| LLaMA 3.2 3B | 0.4503 | 0.2006 | 0.2776 | 151 | 68 | 83 | 271 | 808.64 |
| LLaMA 3.3 | 0.3950 | 0.8378 | 0.5369 | 719 | 284 | 435 | 55 | 4404.18 |
| Mistral Small | 0.5179 | 0.6814 | 0.5885 | 446 | 231 | 215 | 108 | 1663.12 |

## Stage 1+3 (Layout + Conclusion Detection)

| Model | Precision | Recall | F1 Score | Predictions | True Positives | False Positives | False Negatives | Duration (s) |
|-------|-----------|--------|----------|-------------|----------------|-----------------|-----------------|--------------|
| LLaMA 4 Scout | 0.5724 | 0.7463 | 0.6479 | 442 | 253 | 189 | 86 | 0.0 |
| LLaMA 3.1 8B | 0.3293 | 0.7227 | 0.4524 | 744 | 245 | 499 | 94 | 1002.74 |
| LLaMA 3.2 3B | 0.4579 | 0.1445 | 0.2197 | 107 | 49 | 58 | 290 | 916.88 |
| LLaMA 3.3 | 0.3866 | 0.8201 | 0.5255 | 719 | 278 | 441 | 61 | 4073.20 |
| Mistral Small | 0.4979 | 0.7021 | 0.5826 | 478 | 238 | 240 | 101 | 1651.03 |

## Stage 2+3 (Page Classification + Conclusion Detection)

| Model | Precision | Recall | F1 Score | Predictions | True Positives | False Positives | False Negatives | Duration (s) |
|-------|-----------|--------|----------|-------------|----------------|-----------------|-----------------|--------------|
| LLaMA 4 Scout | 0.7470 | 0.7404 | 0.7437 | 336 | 251 | 85 | 88 | 6079.39 |
| LLaMA 3.1 8B | 0.4755 | 0.6578 | 0.5520 | 469 | 223 | 246 | 116 | 2382.82 |
| LLaMA 3.2 3B | 0.7528 | 0.1976 | 0.3131 | 89 | 67 | 22 | 272 | 2219.95 |
| LLaMA 3.3 | 0.5776 | 0.8230 | 0.6788 | 483 | 279 | 204 | 60 | 6457.07 |
| Mistral Small | 0.7551 | 0.6549 | 0.7014 | 294 | 222 | 72 | 117 | 3241.77 |

## Full Pipeline (Layout + Classification + Conclusion Detection)

| Model | Precision | Recall | F1 Score | Predictions | True Positives | False Positives | False Negatives | Duration (s) |
|-------|-----------|--------|----------|-------------|----------------|-----------------|-----------------|--------------|
| LLaMA 4 Scout | 0.7259 | 0.7109 | 0.7183 | 332 | 241 | 91 | 98 | 6532.87 |
| LLaMA 3.1 8B | 0.5096 | 0.7080 | 0.5926 | 471 | 240 | 231 | 99 | 2796.62 |
| LLaMA 3.2 3B | 0.6769 | 0.1298 | 0.2178 | 65 | 44 | 21 | 295 | 8261.13 |
| LLaMA 3.3 | 0.5839 | 0.8112 | 0.6790 | 471 | 275 | 196 | 64 | 6174.56 |
| Mistral Small | 0.7516 | 0.6962 | 0.7228 | 314 | 236 | 78 | 103 | 3272.12 |

## Summary Notes

- **Total Ground Truth Pages**: 339 (consistent across all experiments)
- **Best F1 Score Overall**: LLaMA 4 Scout in Stage 2+3 (0.7437)
- **Best Precision**: LLaMA 4 Scout and Mistral Small both achieve ~0.75 in Stage 2+3 and Full Pipeline
- **Best Recall**: LLaMA 3.3 achieves highest recall (0.8378) in Stage 3 Only
- **Duration**: LLaMA 4 Scout shows 0.0 duration (likely measurement issue), while LLaMA 3.2 3B shows longest duration in Full Pipeline (8261.13s)