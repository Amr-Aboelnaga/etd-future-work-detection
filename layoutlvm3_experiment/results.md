# Results: Header Detection and Classification Performance Analysis

## Executive Summary

This study evaluated multiple approaches for detecting and classifying section headers in academic documents, specifically targeting concluding sections (conclusions, future work, summaries, discussions, recommendations, limitations, and implications). The experiments tested various combinations of confidence thresholds and classification methods on a ground truth dataset of 336 target pages.

**Key Findings:**
- **Best Overall Performance**: Ollama LLM (llama4:scout) achieved the highest F1-score of 0.557
- **Precision Leader**: NLI with 0.8 threshold achieved highest precision of 0.503  
- **Recall Leader**: Lexical matching with partial matching achieved highest recall of 0.866
- **Optimal Trade-off**: Lexical + partial matching at 0.8 threshold (F1: 0.436, Precision: 0.292, Recall: 0.860)

## Detailed Results Analysis

### 1. Lexical Classification Results

The lexical approach using fuzzy string matching showed significant variation based on threshold and matching strategy:

#### Standard Fuzzy Matching (No Partial Matching)
| Threshold | Precision | Recall | F1-Score | True Positives | False Positives | False Negatives |
|-----------|-----------|--------|----------|----------------|-----------------|-----------------|
| 0.75      | 0.280     | 0.062  | 0.102    | 21             | 54              | 315             |
| 0.60      | 0.168     | 0.217  | 0.190    | 73             | 361             | 263             |
| 0.50      | 0.082     | 0.458  | 0.139    | 154            | 1731            | 182             |

#### Partial Matching Mode
| Threshold | Precision | Recall | F1-Score | True Positives | False Positives | False Negatives |
|-----------|-----------|--------|----------|----------------|-----------------|-----------------|
| 0.80      | 0.292     | 0.860  | 0.436    | 289            | 702             | 47              |
| 0.70      | 0.275     | 0.863  | 0.418    | 290            | 763             | 46              |
| 0.60      | 0.243     | 0.866  | 0.380    | 291            | 905             | 45              |

**Analysis:**
- **Dramatic improvement with partial matching**: Recall increases from 0.062-0.458 (standard) to 0.860-0.866 (partial)
- **Threshold effects**: Higher thresholds in both modes generally improve precision
- **Standard mode limitations**: Very low recall (0.062-0.458) makes standard fuzzy matching inadequate
- **Partial matching trade-offs**: Excellent recall but high false positive rates (702-905 FPs)
- **Optimal configuration**: 0.8 threshold with partial matching provides best balance (F1: 0.436)

### 2. Natural Language Inference (NLI) Results

Using the DeBERTa-large-MNLI model for zero-shot classification:

| Confidence Threshold | Precision | Recall | F1-Score | True Positives | False Positives | False Negatives |
|---------------------|-----------|--------|----------|----------------|-----------------|-----------------|
| 0.70                | 0.331     | 0.634  | 0.435    | 213            | 430             | 123             |
| 0.80                | 0.503     | 0.295  | 0.371    | 99             | 98              | 237             |

**Analysis:**
- NLI achieves the highest precision (0.503 at 0.8 threshold)
- Lower threshold (0.7) provides better recall but reduced precision
- F1-scores are competitive with lexical partial matching

### 3. Semantic Similarity Results

Using Sentence-BERT with prototype-based classification:

| Threshold | Precision | Recall | F1-Score | True Positives | False Positives | False Negatives |
|-----------|-----------|--------|----------|----------------|-----------------|-----------------|
| 0.50      | 0.185     | 0.628  | 0.286    | 211            | 929             | 125             |
| 0.60      | 0.284     | 0.333  | 0.306    | 112            | 283             | 224             |
| 0.70      | 0.217     | 0.054  | 0.086    | 18             | 65              | 318             |

**Analysis:**
- Lower thresholds favor recall but compromise precision significantly
- 0.6 threshold provides reasonable balance
- Overall performance is lower than lexical and NLI approaches

### 4. Large Language Model (LLM) Results

Using Ollama with llama4:scout model:

| Model Configuration | Precision | Recall | F1-Score | True Positives | False Positives | False Negatives |
|-------------------|-----------|--------|----------|----------------|-----------------|-----------------|
| llama4:scout      | 0.427     | 0.798  | 0.557    | 268            | 359             | 68              |

**Analysis:**
- **Best overall performance** with highest F1-score
- Excellent recall (0.798) while maintaining reasonable precision
- Demonstrates effectiveness of instruction-tuned language models


## Performance Comparison and Rankings

### By F1-Score (Overall Performance)
1. **Ollama llama4:scout**: 0.557 (Best overall)
2. **Lexical + Partial (0.8)**: 0.436  
3. **NLI (0.7)**: 0.435
4. **Lexical + Partial (0.7)**: 0.418
5. **Lexical + Partial (0.6)**: 0.380
6. **NLI (0.8)**: 0.371
7. **Semantic (0.6)**: 0.306
8. **Semantic (0.5)**: 0.286
9. **Lexical Standard (0.6)**: 0.190
10. **Lexical Standard (0.5)**: 0.139
11. **Lexical Standard (0.75)**: 0.102
12. **Semantic (0.7)**: 0.086

### By Precision (Minimizing False Positives)
1. **NLI (0.8)**: 0.503 (Best precision)
2. **Ollama llama4:scout**: 0.427
3. **Lexical + Partial (0.8)**: 0.292
4. **Semantic (0.6)**: 0.284
5. **Lexical Standard (0.75)**: 0.280
6. **Lexical + Partial (0.7)**: 0.275
7. **Lexical + Partial (0.6)**: 0.243
8. **Semantic (0.7)**: 0.217

### By Recall (Maximizing Coverage)
1. **Lexical + Partial (0.6)**: 0.866 (Best recall)
2. **Lexical + Partial (0.7)**: 0.863
3. **Lexical + Partial (0.8)**: 0.860
4. **Ollama llama4:scout**: 0.798
5. **NLI (0.7)**: 0.634

## Key Insights and Observations

### 1. Method Effectiveness
- **LLM approaches** (Ollama) provide the best balance of precision and recall
- **Lexical with partial matching** achieves excellent recall but moderate precision
- **NLI methods** excel at precision but may miss some target sections
- **Semantic similarity** underperforms compared to other approaches

### 2. Threshold Analysis
- **Higher confidence thresholds** generally improve precision at the cost of recall
- **Partial matching critical**: Enables 13x improvement in lexical recall (0.062→0.860+)
- **Method-specific optimal thresholds**:
  - Lexical partial matching: 0.8 (best F1-score balance)
  - NLI: 0.7 for recall, 0.8 for precision
  - Semantic: 0.6 (higher thresholds cause severe recall drops)
- **Threshold sensitivity**: Semantic methods show extreme sensitivity, with 0.7 threshold causing recall collapse to 0.054

### 3. False Positive Analysis
The high false positive rates (especially in lexical methods) suggest:
- Many detected headers are not actually section headers
- LayoutLMv3 may over-detect headers in academic documents  
- Additional filtering or improved header detection is needed

### 4. False Negative Analysis
Lower false negative rates in partial matching approaches indicate:
- Traditional exact matching misses many valid headers
- Flexible matching strategies are essential for academic documents
- Header text variations require adaptive classification methods

## Statistical Significance

### Dataset Characteristics
- **Total Ground Truth Pages**: 336 target pages
- **Total Predictions Range**: 75-1,196 depending on method
- **Coverage**: All methods tested on identical filtered dataset
- **Consistency**: Page-level evaluation ensures fair comparison

### Performance Variability
- **Precision Range**: 0.082 - 0.503 (6.1x variation)
- **Recall Range**: 0.054 - 0.866 (16x variation)  
- **F1-Score Range**: 0.086 - 0.557 (6.5x variation)
- **Total Predictions Range**: 75 - 1,885 (25x variation across methods)
- **Consistent Ground Truth**: All methods evaluated against the same 336 target pages

## Limitations and Considerations

### 1. Ground Truth Coverage
- Limited to 336 manually annotated pages
- May not represent full diversity of academic document styles
- Potential annotation bias toward certain section types

### 2. Header Detection Dependency
- All classification methods depend on LayoutLMv3 header detection quality
- Detection threshold (0.7) may not be optimal for all document types
- False negatives in detection stage cannot be recovered by classification

### 3. Binary Classification Limitation
- Grouping all target section types may obscure individual section performance
- Some section types may be harder to identify than others
- Future work could benefit from multi-class evaluation

### 4. Computational Requirements
- Multi-GPU setup required for reasonable processing times
- LLM approaches require significant computational resources
- Trade-offs between accuracy and computational efficiency

## Recommendations

### 1. Production Deployment
**Recommendation**: Use Ollama llama4:scout for highest overall performance
- **Rationale**: Best F1-score (0.557) with good precision-recall balance
- **Trade-off**: Higher computational cost but superior accuracy

### 2. High-Precision Applications
**Recommendation**: Use NLI with 0.8 threshold for minimal false positives
- **Rationale**: Highest precision (0.503) reduces manual review burden
- **Trade-off**: Lower recall may miss some target sections

### 3. High-Recall Applications  
**Recommendation**: Use lexical partial matching with 0.6 threshold
- **Rationale**: Highest recall (0.866) ensures comprehensive coverage
- **Trade-off**: High false positive rate requires post-processing

### 4. Balanced Performance
**Recommendation**: Use lexical partial matching with 0.8 threshold
- **Rationale**: Good F1-score (0.436) with excellent recall (0.860)
- **Trade-off**: Moderate precision but computationally efficient

## Future Work Directions

### 1. Improved Header Detection
- Investigate alternative header detection models
- Fine-tune LayoutLMv3 on academic document-specific data
- Explore confidence threshold optimization per document type

### 2. Multi-Class Classification
- Separate evaluation for each target section type
- Investigate section-specific classification strategies
- Analyze performance differences across section types

### 3. Hybrid Approaches
- Ensemble methods combining multiple classification approaches
- Confidence-weighted voting schemes
- Sequential filtering with multiple methods

### 4. Dataset Enhancement
- Expand ground truth annotations
- Include diverse academic disciplines and document formats
- Cross-institutional validation studies

This comprehensive evaluation demonstrates that while significant progress has been made in automated header classification, the choice of method should be guided by specific application requirements, balancing precision, recall, and computational constraints.