# Methodology: Pipeline Experiment for Academic Document Conclusion Detection

## Overview

This experiment compares four pipeline configurations for detecting conclusion and future work sections in academic PDF documents (theses, dissertations, research papers). The study tests four different combinations of processing stages.

## Research Objective

The research question is: **How do different combinations of PDF processing stages (layout extraction, page classification, conclusion detection) perform in identifying conclusion and future work sections in academic documents?**

## Experimental Design

### Pipeline Architecture Overview

Four pipeline configurations were tested:

1. **Stage 3 Only**: Direct conclusion detection on raw PDF text
2. **Stage 1+3**: Layout-preserved text extraction followed by conclusion detection  
3. **Stage 2+3**: Page classification followed by conclusion detection
4. **Full Pipeline**: All three stages (layout extraction → page classification → conclusion detection)

### Detailed Stage Implementations

#### Stage 1: Layout-Preserved Text Extraction

**Technical Implementation**: PyMuPDF (fitz) spatial text processing

**Algorithm Specification**:

1. **Spatial Text Extraction**:
   ```python
   blocks_dict = page.get_text("dict")  # Extract with coordinate metadata
   ```
   - Extracts text in dictionary mode to preserve bounding box coordinates
   - Maintains hierarchical structure: page → blocks → lines → spans
   - Preserves font information, positioning data, and text properties

2. **Coordinate Processing**:
   ```python
   line_x0 = min(span["bbox"][0] for span in line["spans"])  # Leftmost position
   line_y0 = line["bbox"][1]  # Vertical position for sorting
   ```
   - Tracks leftmost x-coordinate for accurate indentation calculation
   - Uses y0 coordinate for vertical positioning and reading order

3. **Reading Order Reconstruction**:
   ```python
   text_lines.sort(key=lambda x: x["y0"])  # Vertical position sorting
   ```
   - Sorts all text lines by vertical position (top to bottom)
   - Maintains natural reading flow regardless of PDF internal structure

4. **Paragraph Detection Algorithm**:
   ```python
   if abs(line["y0"] - last_y) > 10:  # 10-point threshold
       # Start new paragraph
   ```
   - Uses 10-point vertical spacing threshold for paragraph boundaries
   - Groups closely-positioned lines into coherent paragraphs
   - Preserves document logical structure

5. **Indentation Calculation**:
   ```python
   indent_level = int((line["x0"] - min_x) / 10)  # 10 points = 1 indent level
   indent_spaces = " " * indent_level
   ```
   - Calculates relative indentation from leftmost page margin
   - Maps 10 PDF points to 1 space-based indent level
   - Preserves hierarchical document structure (headings, lists, quotes)

6. **Text Reconstruction**:
   ```python
   formatted_text += indent_spaces + line["text"] + "\n"
   # Add paragraph separation
   formatted_text += "\n"  
   ```
   - Rebuilds text with preserved indentation and spacing
   - Maintains visual hierarchy

**Output Characteristics**:
- Preserves document visual hierarchy (indentation, spacing)
- Maintains reading order through coordinate-based sorting
- Retains paragraph structure and formatting cues
- Removes excessive whitespace while preserving meaningful spacing

#### Stage 2: Page Classification

**Technical Implementation**: Large Language Model (LLM) binary classification via Ollama API

**Classification Categories**:
- **Cover page**: Title, author, publication information
- **Table of contents page**: Chapter/section listings with page numbers  
- **Chapter beginning page**: New chapter starts with titles/numbers
- **Normal text**: Regular content pages

**Prompting Strategy**:

**System Message** (342 tokens):
```
You are a PDF page classifier. Analyze the page text and classify it into exactly one of these categories: 
'Cover page', 'Table of contents page', 'Chapter beginning page', or 'Normal text'.

A cover page typically has the title, author, and publication information. 
A table of contents page lists chapters or sections with page numbers. 
A chapter beginning page usually starts with a chapter number or title and may have a drop cap or decorative element.
Normal text pages contain regular paragraphs of content.

Respond with ONLY the category name, nothing else.
```

**User Message Template**:
```
Analyze this page text and classify it:
```
Be conservative - only mark as chapter beginning if you're confident this page starts a new chapter, not just a section within a chapter.

{page_text[:500]}
```
```

**Critical Prompting Design Elements**:
- **Conservative Bias**: Explicit instruction to avoid false positive chapter classifications
- **Text Truncation**: Exactly first 500 characters to focus on page opening content
- **Binary Response**: Constrained output format for deterministic parsing
- **Context Specificity**: Academic document-focused examples and criteria

**API Configuration**:
```python
{
    "model": model_name,
    "messages": [system_message, user_message],
    "stream": False,
    "temperature": 0  # Deterministic output
}
```

**Error Handling**:
- **Retry Mechanism**: 3 attempts with exponential backoff
- **Fallback Strategy**: Defaults to "Normal text" on all failures
- **Timeout Control**: 30-second hard timeout per request

#### Stage 3: Conclusion/Future Work Detection

**Technical Implementation**: LLM-based binary content classification via Ollama API

**Target Content Categories**:
- **Conclusions**: "Conclusions", "Summary and Conclusions", "Final Remarks", "Concluding Remarks"
- **Future Work**: "Future Work", "Future Directions", "Recommendations", "Further Work", "Next Steps"
- **Summaries**: "Summary", "Executive Summary", "Abstract of Findings"
- **Discussions**: "Discussion", "General Discussion", "Interpretation of Results"
- **Recommendations**: "Practical Implications", "Policy Recommendations"
- **Limitations**: "Study Limitations", "Constraints"
- **Implications**: "Theoretical Implications", "Practical Implications"

**Advanced Prompting Strategy**:

**System Message** (1,247 tokens):
```
You are a research paper analyzer. Your task is to determine if a given text is the beginning of a MAIN chapter that covers conclusions, future work, or related content from an academic paper or thesis.

CLASSIFY AS 'Yes' if the text is the beginning of a MAIN chapter covering:
- Conclusions: "Conclusions", "Conclusion", "Summary and Conclusions", "Final Remarks", "Discussion and Conclusions", "Concluding Remarks"
- Future Work: "Future Work", "Future Directions", "Future Research", "Recommendations", "Further Work", "Future Studies", "Next Steps", "Prospects"
- Summaries: "Summary", "Executive Summary", "Abstract of Findings"
- Discussions: "Discussion", "General Discussion", "Interpretation of Results"
- Recommendations: "Recommendations", "Practical Implications", "Policy Recommendations"
- Limitations: "Limitations", "Study Limitations", "Constraints"
- Implications: "Implications", "Theoretical Implications", "Practical Implications"

CLASSIFY AS 'No' for:
- Subsections within other chapters (e.g., "3.7 Conclusions" within a results chapter)
- Introduction, methodology, results, literature review, or other standard academic sections
- Brief concluding paragraphs that are not standalone chapters
- Normal body text with references or citations
- Table of contents, list of figures, list of tables
- Bibliographies, reference lists, appendices
- Author information, acknowledgments, dedications
- Page headers, footers, or page numbers
- Abstract or executive summary of the entire document
- Random text fragments or partial sentences

IMPORTANT: We only want MAIN, STANDALONE chapters or major sections, NOT subsections, navigation elements, or body text.

Respond with ONLY 'Yes' or 'No'.
```

**User Message Template**:
```
Analyze this text and determine if it's the beginning of a main chapter covering conclusions, future work, summaries, discussions, recommendations, limitations, or implications:

```
{content_excerpt}
```

Is this the beginning of a main chapter in one of these categories?

Respond with ONLY 'Yes' or 'No'.
```

**Prompting Design Elements**:
- **Explicit Negative Examples**: List of content to exclude
- **Hierarchical Distinction**: Emphasizes "MAIN" vs. subsection differentiation  
- **Conservative Classification**: Instructions to minimize false positives
- **Academic Domain Specificity**: Examples from scholarly document structure
- **Binary Response Constraint**: Yes/No classification only

### Critical Prompting Strategy Differences

#### Direct Conclusion Detection (Stage 3 Only, Stage 1+3)

**Key Difference**: Single-step prompts that must distinguish conclusion content from ALL other page types.

**Stage 3 User Message** (Used in Stage 3 Only and Stage 1+3):
```
Analyze this page text and classify it:
```
Be conservative - only mark as chapter beginning if you're confident this page starts a new chapter, not just a section within a chapter. Only mark as true if you're confident this is the actual start of a Future Work or Conclusions chapter, not just a mention of future work or conclusions within another chapter.

{page_text[:500]}
```
```

**Critical Characteristics**:
- **Dual Task**: Must identify both chapter beginnings AND conclusion content simultaneously
- **Additional Context**: Includes extra instruction about "Future Work or Conclusions chapter"
- **Higher Cognitive Load**: Single prompt handles two distinct classification tasks
- **Broad Discrimination**: Must distinguish conclusions from covers, TOCs, normal text, etc.

#### Two-Step Classification (Stage 2+3, Full Pipeline)

**Key Difference**: Specialized prompts where page classification filters before conclusion detection.

**Stage 2 User Message** (Page Classification):
```
Analyze this page text and classify it:
```
Be conservative - only mark as chapter beginning if you're confident this page starts a new chapter, not just a section within a chapter.

{page_text[:500]}
```
```

**Stage 3 User Message** (Conclusion Detection on Pre-filtered Pages):
```
Analyze this text and determine if it's the beginning of a main chapter covering conclusions, future work, summaries, discussions, recommendations, limitations, or implications:

```
{content_excerpt}
```

Is this the beginning of a main chapter in one of these categories?

Respond with ONLY 'Yes' or 'No'.
```

**Critical Characteristics**:
- **Specialized Tasks**: Each prompt optimized for single classification task
- **Reduced Context**: Stage 3 prompt assumes input is already a chapter beginning
- **Lower Cognitive Load**: Each step focuses on one decision criterion
- **Narrow Discrimination**: Stage 3 only distinguishes conclusion types from other chapter types

### Prompting Strategy Impact Analysis

#### Cognitive Load Distribution

**Direct Detection (Single-Step)**:
- **High Load**: "Is this a chapter beginning AND is it a conclusion chapter?"
- **Complex Decision**: Multiple criteria evaluated simultaneously
- **Context Confusion**: May conflate chapter detection with conclusion detection

**Two-Step Classification**:
- **Distributed Load**: "Is this a chapter beginning?" → "Is this chapter about conclusions?"
- **Sequential Decisions**: Each step has clear, focused criteria
- **Specialized Context**: Each prompt optimized for its specific task

#### Error Propagation Patterns

**Direct Detection**:
- **Error Type**: False positives from normal text mentioning "conclusions"
- **Failure Mode**: Confuses conclusion mentions within chapters with conclusion chapters
- **Precision Impact**: High false positive rate on non-chapter pages

**Two-Step Classification**:
- **Error Type**: False negatives from page classification errors
- **Failure Mode**: Conclusion chapters missed if not classified as "chapter beginning"
- **Recall Impact**: Potential recall reduction but dramatic precision improvement

#### Prompt Engineering Sophistication

**Direct Detection Prompts**:
- **Complexity**: Attempts to handle both tasks in single prompt
- **Instruction Density**: Heavy cognitive load with multiple criteria
- **Error Handling**: Relies on conservative bias to reduce false positives

**Two-Step Classification Prompts**:
- **Specialization**: Each prompt optimized for specific task
- **Instruction Clarity**: Clear, focused decision criteria per step
- **Error Mitigation**: Page filtering dramatically reduces conclusion detection noise

### Pipeline-Specific Processing Differences

#### Stage 3 Only Pipeline

**Text Processing**:
- **Extraction Method**: Basic `page.get_text()` without layout preservation
- **Page Scope**: ALL pages with ≥50 characters processed
- **Processing Flow**: Direct conclusion detection on every qualifying page

**Prompting Strategy**:
- **Single-Step**: Direct conclusion detection without preliminary classification
- **Dual-Task Prompts**: Must identify chapter beginnings AND conclusion content simultaneously
- **High Cognitive Load**: Single prompt handles complex multi-criteria decision
- **Broad Discrimination**: Must distinguish conclusions from covers, TOCs, normal text, etc.
- **Raw Text Input**: Unprocessed PDF text may lack formatting context

**Characteristics**:
- Processes all pages meeting minimum length requirement
- Single-step processing
- No page filtering

#### Stage 1+3 Pipeline

**Text Processing**:
- **Extraction Method**: Enhanced `_extract_text_with_layout()` with spatial processing
- **Page Scope**: ALL pages with ≥50 characters processed
- **Processing Flow**: Layout preservation → conclusion detection (skips page classification)

**Prompting Strategy**:
- **Single-Step**: Direct conclusion detection with layout-enhanced input
- **Dual-Task Prompts**: Same complex multi-criteria decision as Stage 3 Only
- **Enhanced Context**: Indentation and structure cues available to LLM
- **Spatial Awareness**: Document formatting preserved
- **Same Cognitive Load**: Still requires simultaneous chapter + conclusion identification

**Characteristics**:
- Processes all pages meeting minimum length requirement
- Uses layout-preserved text
- Single-step processing with enhanced text input

#### Stage 2+3 Pipeline

**Text Processing**:
- **Extraction Method**: Basic `page.get_text()` without layout preservation
- **Page Scope**: ONLY pages classified as "Chapter beginning page"
- **Processing Flow**: Page classification → conclusion detection on filtered pages

**Prompting Strategy**:
- **Two-Step Process**: Specialized prompts for page classification → conclusion detection
- **Task Decomposition**: Separates chapter identification from conclusion identification
- **Lower Cognitive Load**: Each prompt focused on single decision criterion
- **Precision Focus**: Stage 3 only processes pre-filtered chapter beginnings
- **Specialized Conclusion Prompt**: Assumes input is already a chapter, focuses on content type
- **Page Filtering**: Page classification reduces conclusion detection noise

**Characteristics**:
- Processes only pages classified as "Chapter beginning page"
- Two-step processing
- Reduced number of pages processed by Stage 3

#### Full Pipeline (All Stages)

**Text Processing**:
- **Extraction Method**: Enhanced `_extract_text_with_layout()` with spatial processing
- **Page Scope**: ONLY pages classified as "Chapter beginning page"  
- **Processing Flow**: Layout preservation → page classification → conclusion detection

**Prompting Strategy**:
- **Two-Step Enhanced**: Specialized prompts with layout-aware text processing
- **Task Decomposition**: Separates chapter identification from conclusion identification
- **Layout-Enhanced Prompts**: Both steps benefit from preserved document structure
- **Combined Features**: Structural filtering with enhanced text representation
- **Multiple Components**: Specialized prompts + layout preservation + page filtering
- **Task-Specific Prompts**: Each prompt focused on its specific task

**Characteristics**:
- Processes only pages classified as "Chapter beginning page"
- Uses layout-preserved text
- Two-step processing with enhanced text input
- Highest computational cost

### Language Models Evaluated

Five language models were tested:

1. **LLaMA 4 Scout**: Model variant
2. **LLaMA 3.1 8B**: 8-billion parameter model
3. **LLaMA 3.2 3B**: 3-billion parameter model
4. **LLaMA 3.3**: Model variant  
5. **Mistral Small**: Compact model

**Model Access**: All models served via Ollama API with the same configuration parameters.

### Ground Truth Dataset

#### Dataset Composition
- **Source File**: `final_groundtruth_filtered.csv`
- **Total Entries**: 339 manually labeled conclusion pages across 222 unique PDFs
- **Format**: CSV with structured metadata

**Exact Schema**:
```
full_path,department,college,page_number,chapter_title,similarity_score,matched_query,source,label
```

**Critical Fields**:
- **`full_path`**: Absolute filesystem path for file matching
- **`page_number`**: Specific page containing conclusion content (1-indexed)
- **`chapter_title`**: Actual chapter title (e.g., "CONCLUSIONS", "Chapter 8. Summary and Future Work")
- **`label`**: Content classification (`CONCLUSION`, `FUTURE_WORK`, `SUMMARY`, combinations like `CONCLUSION|FUTURE_WORK`)
- **`source`**: Labeling origin (`manual_labels`, `toc_analysis`)

#### Validation Process
- **Manual Verification**: Domain expert validation of all conclusion page labels
- **Quality Control**: Multiple-source labeling with consistency checking
- **Academic Diversity**: Multiple institutions, departments, and document types

### Experimental Controls

#### Technical Control Parameters

**LLM API Configuration**:
```python
api_config = {
    "temperature": 0,           # Deterministic output
    "stream": False,           # Synchronous processing
    "timeout": 30,             # 30-second request limit
    "max_retries": 3,          # Triple retry mechanism
    "concurrent_limit": 8      # Maximum simultaneous requests
}
```

**Text Processing Standards**:
- **Input Truncation**: Exactly 500 characters for all LLM analysis
- **Minimum Content**: Pages <50 characters excluded from processing
- **Encoding**: UTF-8 throughout entire pipeline
- **Spacing Normalization**: Excessive newlines collapsed via `re.sub(r'\n{3,}', '\n\n', text)`

**Layout Algorithm Parameters**:
- **Indentation Mapping**: 10 PDF points → 1 space character
- **Paragraph Threshold**: 10-point vertical spacing for paragraph breaks
- **Coordinate Precision**: Float precision maintained throughout processing
- **Reading Order**: Strict y-coordinate ascending sort

#### Data Control Mechanisms

**Subset Evaluation Strategy**:
- **Filtered Comparison**: Only evaluates PDFs present in ground truth subset
- **Exact Matching**: Binary classification based on (PDF path, page number) tuples
- **False Positive Prevention**: Predictions for non-ground-truth PDFs ignored

**Consistency Measures**:
- **Same Ground Truth**: Identical labeled dataset across all experiments
- **Deterministic Processing**: Fixed random seeds and API temperature settings
- **Environmental Control**: Consistent hardware and software configurations

#### Infrastructure Control

**Processing Architecture**:
- **Parallel Execution**: ProcessPoolExecutor with controlled worker counts
- **Progress Tracking**: JSON-based resumable operations (disabled for full re-processing)
- **Error Isolation**: Individual PDF failures don't terminate batch processing

**API Load Management**:
- **Connection Pooling**: `aiohttp.TCPConnector` with 2× concurrent request limits
- **Semaphore Control**: `asyncio.Semaphore` for precise concurrency management
- **Circuit Breaking**: Automatic fallback on repeated API failures

### Evaluation Methodology

#### Primary Metrics
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)  
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)

#### Supporting Metrics
- **True Positives (TP)**: Correctly identified conclusion pages
- **False Positives (FP)**: Incorrectly identified pages as conclusions
- **False Negatives (FN)**: Missed actual conclusion pages
- **Processing Duration**: Wall-clock time for complete pipeline execution

#### Evaluation Implementation
```python
def calculate_metrics(predictions, ground_truth):
    """
    Binary classification evaluation with exact page matching
    """
    tp = len(predictions & ground_truth)      # Set intersection
    fp = len(predictions - ground_truth)      # Predicted but not actual
    fn = len(ground_truth - predictions)      # Actual but not predicted
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1
```

### Reproducibility Framework

#### Version Control
- **Code Versioning**: All processor scripts version-controlled
- **Dependency Tracking**: Exact package versions documented
- **Model Versions**: Specific model checkpoints and serving configurations

#### Execution Documentation
- **Command Logging**: Complete execution commands with parameters
- **Timestamped Results**: All outputs stored in timestamped directories
- **Configuration Preservation**: API endpoints, model names, and parameters recorded

#### Data Integrity
- **Hash Verification**: PDF files and ground truth data checksummed
- **Path Validation**: All file paths verified before processing
- **Output Validation**: Results format checking and consistency verification

### Limitations and Assumptions

#### Technical Limitations
- **Random Seed Control**: LLM output randomness not explicitly controlled beyond temperature=0
- **API Dependency**: Results dependent on Ollama service availability and consistency
- **Single Language**: Only English-language academic documents processed
- **PDF Format**: Assumes text-based PDFs, not scanned images

#### Methodological Limitations  
- **Ground Truth Scale**: Limited to 339 labeled pages (potential dataset bias)
- **Manual Annotation**: Human subjectivity in ground truth creation
- **Binary Classification**: Conclusion detection treated as binary rather than confidence-based
- **Page Granularity**: No sub-page location analysis or confidence scoring

#### Statistical Assumptions
- **Independence**: Assumes statistical independence between PDF documents
- **Sample Representativeness**: Ground truth assumed representative of broader academic corpus
- **Evaluation Bias**: Same dataset used for development and evaluation (no held-out test set)

### Experimental Validity

#### Internal Validity
- **Controlled Variables**: Consistent API parameters, text processing, and evaluation metrics
- **Systematic Variation**: Only pipeline configuration varies between conditions
- **Measurement Consistency**: Identical evaluation procedures across all experiments

#### External Validity
- **Document Diversity**: Multiple academic institutions, departments, and document types
- **Real-World Relevance**: Actual academic documents rather than synthetic test cases
- **Scale Validation**: Multiple subset sizes demonstrate consistent performance patterns

This methodology describes the experimental framework used to compare pipeline configurations for academic document conclusion detection.