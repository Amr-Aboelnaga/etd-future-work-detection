# Methodology: GROBID-Based Academic Header Detection and Classification

## Overview

This experiment evaluates a GROBID-based approach for detecting and classifying section headers in academic documents, specifically focusing on identifying conclusion, future work, summary, discussion, recommendations, limitations, and implications sections. The methodology employs a two-stage pipeline: header detection using GROBID's document structure parsing capabilities, followed by multi-method classification using lexical matching, Natural Language Inference (NLI), semantic similarity, and Large Language Model (LLM) approaches.

## Dataset and Ground Truth

### Ground Truth Processing Pipeline

The experiment uses a multi-stage ground truth preparation process:

1. **Original Dataset**: Started with `final_groundtruth.csv` containing academic document annotations
2. **Filtering Process** (`filter_groundtruth_by_results.py`): 
   - Filters ground truth to only include PDFs with corresponding GROBID JSON results
   - Maps PDF full paths to ensure data consistency
   - Removes entries where header detection failed
   - Final filtered dataset: `final_groundtruth_filtered.csv` with 336 pages

3. **Coverage Verification** (`check_ground_truth_coverage.py`):
   - Validates that header detection results cover ground truth pages
   - Analyzes detection coverage ratios per PDF
   - Identifies missed target pages for quality assessment

**Ground Truth Labels:**
- `CONCLUSION`: Conclusion sections and final remarks
- `FUTURE_WORK`: Future work and research directions
- `SUMMARY`: Summary sections and executive summaries  
- `DISCUSSION`: Discussion and interpretation sections
- `RECOMMENDATIONS`: Recommendations and practical implications
- `LIMITATIONS`: Study limitations and constraints
- `IMPLICATIONS`: Theoretical and practical implications

**Data Quality Assurance:**
- Cross-validation between PDF paths and detection results
- Coverage analysis to ensure detection completeness
- Statistical validation of ground truth distribution

## Stage 1: Header Detection with GROBID

### GROBID Setup and Processing (`grobid_header_detection.py`)

The header detection stage uses GROBID (GeneRation Of Bibliographic Data), a machine learning library for extracting and parsing bibliographic information from scholarly documents.

**Key Components:**

1. **GROBIDHeaderDetector Class**: Manages connection to GROBID server and processes PDFs
   - Server endpoint: `http://localhost:8070/api`
   - Uses `processFulltextDocument` API with header consolidation enabled
   - Extracts TEI XML format with coordinate information
   - Timeout handling: 5-minute processing timeout per document

2. **TEI XML Parsing**: 
   - Extracts `<head>` elements from document structure (body and front matter)
   - Maps headers to PDF pages using GROBID coordinate system
   - Coordinate format: "page,x,y,width,height" converted to bbox format
   - Filters substantial headers (≥2 words or section markers)
   - Assigns confidence scores based on text length and coordinate presence

3. **Section Marker Detection**:
   - Identifies numbered sections (e.g., "2.1", "Chapter 1")
   - Recognizes Roman numerals and standard academic patterns
   - Regex patterns for common section formatting

4. **Multi-process Architecture**:
   - Parallel processing across 4 workers by default
   - Processes unique PDFs from ground truth dataset
   - Error handling for failed GROBID processing
   - Outputs structured JSON results for each PDF


## Stage 2: Header Classification

The classification stage implements multiple approaches to identify target section types from detected headers.

### Method 1: Lexical Classification (`classify_headers.py`)

**Fuzzy String Matching Approach:**

1. **Target Lexicon**: Combined lexicon of 53 target terms covering all section types:
   ```python
   TARGET_LEX = {
       "conclusion", "conclusions", "future work", "future directions",
       "summary", "discussion", "recommendations", "limitations",
       "implications", "practical implications", ...
   }
   ```

2. **Normalization**: Headers normalized using regex patterns:
   ```python
   def normalize_header_text(s: str) -> str:
       return re.sub(r"\s+", " ", re.sub(r"[^\w\s\-&]", " ", s.lower())).strip()
   ```

3. **Classification Parameters**:
   - **Fuzzy Threshold**: 50, 60, 70, 80 (RapidFuzz ratio scoring)
   - **Partial Matching**: Optional substring matching
   - **Binary Output**: "Relevant" vs "Not Relevant"

4. **Variants Tested**:
   - Standard fuzzy matching at different thresholds
   - Partial matching (substring detection)
   - Combined approaches

### Method 2: Natural Language Inference (`classify_headers.py`)

**Zero-shot Classification Approach:**

1. **Model**: Microsoft DeBERTa-large-MNLI
2. **Hypothesis Template**: "This header is a chapter title about {}."
3. **Candidate Labels**:
   - "a conclusion, summary, discussion, future work, recommendations, limitations, or implications section"
   - "an introduction, methodology, or random text"

4. **Multi-GPU Processing**:
   - Distributed across 6 GPUs
   - Batch processing (batch size 16)
   - Threshold-based binary classification (threshold = 0.8)

### Method 3: Semantic Similarity (`improved_classifier.py`)

**Sentence Transformer Approach:**

1. **Model**: all-MiniLM-L6-v2 sentence transformer
2. **Prototype Embeddings**: 38 representative sentences covering target sections:
   ```python
   RELEVANT_PROTOTYPES = [
       "In conclusion, this study demonstrates",
       "Future research directions include",
       "Summary of findings presented here",
       "General discussion of results",
       ...
   ]
   ```

3. **Classification Process**:
   - Encode headers using sentence transformer
   - Compute cosine similarity with prototype embeddings
   - Apply threshold-based binary classification
   - Tested thresholds: 0.5, 0.6, 0.7, 0.8

### Method 4: Large Language Model (`llm_classify_headers.py`)

**Ollama-based Classification:**

1. **Model**: Llama4:scout via Ollama
2. **Prompt Engineering**: Structured prompts with examples and constraints
3. **Multi-GPU Distribution**: Across 6 Ollama servers (ports 11434-11439)
4. **Response Parsing**: Extracts binary labels and confidence scores

## Evaluation Methodology

### Page-Level Binary Evaluation

The evaluation treats each detected header as a page-level prediction, comparing against ground truth page numbers.

**Metrics Calculation:**
1. **True Positives (TP)**: Pages with both predicted "Relevant" headers and ground truth target sections
2. **False Positives (FP)**: Pages with predicted "Relevant" headers but no ground truth target sections  
3. **False Negatives (FN)**: Pages with ground truth target sections but no predicted "Relevant" headers

**Standard Metrics:**
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)  
- F1-Score = 2 × (Precision × Recall) / (Precision + Recall)

### Cross-Method Comparison

All methods output standardized JSON format enabling direct comparison:
- Consistent evaluation pipeline (`evaluate_all_results.py`)
- Automated metrics collection (`classification_metrics.csv`, `semantic_metrics.csv`)
- Statistical significance testing across approaches

### Comprehensive Evaluation Framework (`evaluate_all_results.py`)

The evaluation system automatically processes all classification result files:

**Multi-File Processing:**
- Scans directory for JSON result files using pattern matching
- Supports multiple naming conventions (lexical*, nli*, semantic*, etc.)
- Infers classification method from filename and data content
- Batch evaluation across all methods for comparison

**Method Detection:**
- Filename pattern recognition for automatic method identification
- Fallback to `classification_method` field extraction
- Support for custom method naming conventions

## Experimental Controls

1. **Ground Truth Filtering**: Only includes PDFs with successful GROBID processing
2. **Consistent Evaluation**: Same ground truth and metrics across all methods
3. **Reproducible Parameters**: Fixed random seeds and model configurations
4. **Ablation Studies**: Multiple parameter settings per method
5. **Coverage Analysis**: Verification of header detection completeness

## Technical Infrastructure

- **GROBID Version**: Latest stable release with TEI coordinate support
- **Hardware**: Multi-GPU setup for parallel processing (6 GPUs for NLI/LLM)
- **Languages**: Python 3.8+ with transformers, sentence-transformers, rapidfuzz
- **Dependencies**: PyMuPDF, pandas, numpy, tqdm, lxml (see `requirements.txt`)
- **Output Formats**: JSON for interoperability, CSV for analysis

### Resource Requirements

**GROBID Processing:**
- Single GROBID server instance
- 4-worker parallel processing
- 5-minute timeout per document
- Memory: Moderate (document parsing)

**Classification Methods:**
- **Lexical**: CPU-only, minimal memory
- **NLI**: 6 GPUs, high memory, batch processing
- **Semantic**: Optional GPU, moderate memory  
- **LLM**: 6 Ollama servers (ports 11434-11439), very high memory

### Automated Metrics Collection

**CSV Export Systems:**
- `classification_metrics.csv`: Lexical and NLI results with timestamps
- `semantic_metrics.csv`: Semantic similarity results
- Standardized column format for cross-method comparison
- Automatic parameter tracking (thresholds, models, batch sizes)

### File Naming Conventions

**Result Files:**
- Lexical: `classified_results_lexical_fuzzy[threshold]_[partial]_[timestamp].json`
- NLI: `classified_results_nli_[model]_[timestamp].json`  
- Semantic: `semantic_results_semantic_[model]_thresh[threshold]_[timestamp].json`
- LLM: `grobid_llama4.json` (custom naming)

This methodology provides a comprehensive evaluation framework comparing traditional lexical approaches with modern NLP methods for academic document structure analysis, with robust infrastructure for reproducible research and systematic comparison across multiple approaches.