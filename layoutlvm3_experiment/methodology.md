# Methodology: Header Detection and Classification in Academic Documents

## Overview

This experiment investigates automated detection and classification of section headers in electronic thesis and dissertation (ETD) documents. The primary objective is to identify concluding sections (conclusions, future work, summaries, discussions, recommendations, limitations, and implications) within academic documents using a two-stage approach: header detection followed by header classification.

## Experimental Design

### Stage 1: Header Detection

The first stage employs a fine-tuned LayoutLMv3 model to detect potential section headers within PDF documents.

#### Model Configuration
- **Base Model**: `Mit1208/layoutlmv3-finetuned-DocLayNet` - A LayoutLMv3 model fine-tuned on the DocLayNet dataset for document layout analysis
- **Target Label**: Section headers (specifically the "section-header" class from DocLayNet)
- **Input Processing**: 
  - PDFs converted to images at 144 DPI
  - Text extraction with word-level bounding boxes using PyMuPDF
  - Bounding boxes normalized to 0-1000 scale for LayoutLMv3
  - Maximum token limit: 512 tokens per page

#### Multi-GPU Processing
- **Implementation**: `multi_gpu_header_detection.py`
- **Parallelization**: Multiple GPU setup (6 GPUs) with process-based parallelization
- **GPU Assignment**: Round-robin distribution of PDFs across available GPUs
- **Process Isolation**: Each worker process uses a dedicated GPU to avoid memory conflicts

#### Header Grouping Algorithm
Adjacent header words are grouped into complete header titles using spatial proximity:
- **Default Line Tolerance**: 20 pixels (function default) / 100 pixels (command-line default)
- **Default Word Gap Tolerance**: 50 pixels (function default) / 100 pixels (command-line default)  
- **Reading Order**: Headers sorted top-to-bottom, left-to-right
- **Multiline Support**: Optional cross-line grouping for multi-line headers
- **Grouping Strategy**: 
  - Same-line grouping: vertical distance ≤ line_tolerance AND horizontal gap ≤ word_gap_tolerance
  - Multi-line grouping: only vertical distance ≤ line_tolerance (ignores horizontal gaps)
- **Confidence Calculation**: Length-weighted average of individual word confidences

#### Confidence Thresholding
- **Primary Threshold**: 0.7 (confidence score for header detection)
- **Alternative Thresholds**: 0.5, 0.6, 0.8 tested for sensitivity analysis

### Stage 2: Header Classification

Detected headers are classified using multiple complementary approaches to identify target section types.

#### Target Section Types
The system identifies headers belonging to:
- **CONCLUSION**: Conclusions, concluding remarks, final remarks, closing thoughts
- **FUTURE_WORK**: Future work, future directions, future research, next steps
- **SUMMARY**: Summary, executive summary, abstract of findings  
- **DISCUSSION**: Discussion, general discussion, interpretation of results
- **RECOMMENDATIONS**: Recommendations, practical implications, policy recommendations
- **LIMITATIONS**: Limitations, study limitations, constraints
- **IMPLICATIONS**: Implications, theoretical implications, practical implications

#### Classification Methods

##### 1. Lexical Classification (`classify_headers.py`)
- **Approach**: Fuzzy string matching against curated lexicon
- **Matching Algorithm**: RapidFuzz ratio scoring  
- **Thresholds Tested**: 0.5, 0.6, 0.7, 0.75, 0.8 fuzzy similarity
- **Partial Matching**: Optional substring matching for improved recall
- **Lexicon**: 42 carefully curated target section phrases covering:
  - Conclusions (9 phrases): "conclusion", "conclusions", "concluding remarks", etc.
  - Future Work (9 phrases): "future work", "future directions", "next steps", etc.
  - Summary (4 phrases): "summary", "executive summary", etc.
  - Discussion (3 phrases): "discussion", "general discussion", etc.
  - Recommendations (4 phrases): "recommendations", "practical implications", etc.
  - Limitations (2 phrases): "limitations", "limitations and future work"
  - Implications (3 phrases): "implications", "theoretical implications", etc.
- **Text Normalization**: Lowercase conversion, punctuation removal, whitespace normalization
- **Binary Output**: "Relevant" vs "Not Relevant"

##### 2. Semantic Classification (`improved_classifier.py`)
- **Model**: Sentence-BERT (`all-MiniLM-L6-v2`) - 384-dimensional embeddings
- **Approach**: Cosine similarity against prototype sentences
- **Prototypes**: 46 representative prototype sentences covering:
  - Conclusion prototypes (12): "In conclusion, this study demonstrates", "Our findings suggest that", etc.
  - Future Work prototypes (12): "Future research directions include", "We plan to extend this work", etc.
  - Summary prototypes (4): "Summary of findings presented here", "This executive summary", etc.
  - Discussion prototypes (4): "General discussion of results", "We discuss the implications", etc.
  - Recommendations prototypes (4): "Our recommendations for practice", "Policy recommendations", etc.
  - Limitations prototypes (3): "Limitations of this study include", "Several limitations", etc.
  - Implications prototypes (3): "Theoretical implications", "Practical implications", etc.
- **Similarity Computation**: Batch processing with cosine similarity matrix operations
- **Thresholds Tested**: 0.3, 0.5, 0.6, 0.7 cosine similarity
- **Processing**: Efficient batch encoding with numpy vectorization
- **Binary Output**: "Relevant" vs "Not Relevant"

##### 3. Natural Language Inference (`classify_headers.py`)
- **Model**: `microsoft/deberta-large-mnli`
- **Approach**: Zero-shot classification with hypothesis templates
- **Hypothesis Template**: "This header is a chapter title about {}."
- **Candidate Labels**: 
  - "a conclusion, summary, discussion, future work, recommendations, limitations, or implications section"
  - "an introduction, methodology, or random text"
- **Multi-GPU Processing**: Distributed across 6 GPUs
- **Confidence Threshold**: 0.7-0.8
- **Binary Output**: "Relevant" vs "Not Relevant"

##### 4. Large Language Model Classification (`llm_classify_headers.py`)
- **Platform**: Ollama with local inference servers
- **Model**: `llama4:scout` (instruction-tuned variant)
- **Multi-Server Architecture**: 
  - 6 Ollama servers on ports 11434-11439 (one per GPU)
  - Process-based parallelization with chunk distribution
  - Base port 11434 + gpu_id for server assignment
- **Prompt Engineering**: 
  - Structured system message with detailed instructions
  - Task-specific hypothesis: "This header is a chapter title about {}"
  - Explicit target section enumeration and examples
  - Constraint: "WE ARE ONLY INTERESTED IN CHAPTER HEADERS OR FULL SECTION HEADERS not SUBSECTION HEADERS"
- **Request Configuration**:
  - Temperature: 0.0 (deterministic responses)
  - Max tokens: 50 (concise responses)
  - Timeout: 60 seconds (with 90-second retry)
  - Rate limiting: 0.1-0.2 second delays between requests
- **Response Processing**:
  - Regex-based parsing with multiple pattern matching
  - Fallback classification based on keyword indicators
  - Error handling with default "Not Relevant" classification
- **Parallel Processing**: ThreadPoolExecutor with max 2 workers per server
- **Binary Output**: "Relevant" vs "Not Relevant" with confidence scores

## Evaluation Framework

### Ground Truth Data
- **Source**: Manually annotated dataset (`final_groundtruth_filtered.csv`)
- **Original Dataset**: `final_groundtruth.csv` (pre-filtering)
- **Coverage**: 336 target pages across multiple PDFs
- **Annotation Granularity**: Page-level annotations with section type labels
- **Filtering Process**: Automated filtering (`filter_groundtruth_by_results.py`) to include only PDFs with corresponding JSON detection results
- **Data Structure**: CSV with columns for full_path, page_number, chapter_title, label, department, college
- **Quality Assurance**: Coverage validation (`check_ground_truth_coverage.py`) to verify detection results match ground truth entries

### Evaluation Metrics
- **Level**: Page-level evaluation (each predicted page counts as one prediction)
- **Binary Classification**: All target section types grouped as "Relevant"
- **Metrics Calculated**:
  - **Precision**: True Positives / (True Positives + False Positives)
  - **Recall**: True Positives / (True Positives + False Negatives)  
  - **F1-Score**: Harmonic mean of precision and recall
  - **True Positives**: Pages correctly identified as containing target sections
  - **False Positives**: Pages incorrectly identified as containing target sections
  - **False Negatives**: Target pages missed by the system

### Evaluation Process
1. **Header Detection**: Extract all headers from PDFs using LayoutLMv3 (`multi_gpu_header_detection.py`)
2. **Result Loading**: Load JSON detection results (`load_pdf_results()` function)
3. **Classification**: Apply classification method to all detected headers:
   - Lexical: Fuzzy matching with configurable thresholds and partial matching
   - Semantic: Batch sentence embedding and cosine similarity computation  
   - NLI: Multi-GPU zero-shot classification with hypothesis templates
   - LLM: Distributed Ollama inference with structured prompting
4. **Page Mapping**: Map classified "Relevant" headers to their respective pages
5. **Ground Truth Comparison**: 
   - Load filtered ground truth annotations
   - Map PDF paths between detection results and annotations
   - Identify target pages (containing CONCLUSION, FUTURE_WORK, SUMMARY, DISCUSSION, RECOMMENDATIONS, LIMITATIONS, or IMPLICATIONS)
6. **Metric Calculation**: Compute precision, recall, and F1-score using set operations
7. **Comprehensive Evaluation**: Multi-method comparison using `evaluate_all_results.py`

## Dataset and Preprocessing

### Document Collection
- **Source**: Electronic Theses and Dissertations (ETDs)
- **Format**: PDF documents with varying layouts and styles
- **Processing**: Multi-GPU distributed processing for scalability
- **Coverage**: Documents filtered to match available ground truth annotations

### Preprocessing Pipeline
1. **PDF Parsing**: Extract text and layout information using PyMuPDF
2. **Image Conversion**: Convert pages to RGB images at 144 DPI
3. **Text Extraction**: Word-level text extraction with bounding box coordinates
4. **Normalization**: Coordinate normalization for LayoutLMv3 input requirements
5. **Tokenization**: LayoutLMv3 tokenizer with truncation and padding

## Experimental Conditions

### Hardware Configuration
- **GPUs**: 6 CUDA-enabled GPUs for parallel processing
- **Memory Management**: Process-based isolation to prevent GPU memory conflicts
- **Load Balancing**: Round-robin GPU assignment for balanced workload

### Software Dependencies
- **Core Libraries**: PyTorch, Transformers, PyMuPDF, Pillow
- **NLP Libraries**: Sentence-transformers, datasets, rapidfuzz
- **Processing**: tqdm for progress tracking, multiprocessing for parallelization

### Hyperparameter Settings
- **Detection Confidence**: 0.7 (primary), with ablation at 0.5, 0.6, 0.8
- **Semantic Similarity**: 0.3-0.7 thresholds tested
- **NLI Confidence**: 0.7-0.8 thresholds evaluated
- **Batch Sizes**: 16 for NLI processing, optimized for GPU memory
- **Header Grouping**: 20px line tolerance, 50px word gap tolerance

## Quality Assurance

### Data Validation
- **Ground Truth Filtering**: Automatic filtering to match processed documents
- **Coverage Analysis**: Systematic verification of detection coverage
- **Error Handling**: Robust exception handling and logging
- **Results Validation**: Cross-validation between different methods

### Reproducibility Measures
- **Random Seeds**: Fixed seeds for deterministic results
- **Process Isolation**: Independent worker processes to prevent interference
- **Detailed Logging**: Comprehensive logging of processing parameters
- **Result Serialization**: JSON output format with complete metadata

This methodology ensures a systematic and rigorous evaluation of header detection and classification approaches for academic document processing, with particular focus on identifying concluding sections that are crucial for document understanding and information extraction.