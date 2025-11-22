# Enhanced PDF Processor

A comprehensive tool for processing academic PDFs (theses, dissertations, research papers) through a three-stage pipeline that extracts, classifies, and analyzes document structure with a focus on identifying chapter beginnings and conclusion sections.

## Overview

The Enhanced PDF Processor combines advanced PDF parsing, machine learning-based page classification, and specialized content analysis into a single streamlined pipeline. It's designed to handle large collections of academic documents and systematically extract structural information for research purposes.

## Three-Stage Pipeline Architecture

### Stage 1: PDF Parsing with Layout Preservation

**Purpose**: Extract text from PDF pages while maintaining the original document layout, formatting, and spatial relationships.

**Technical Implementation**:
- Uses PyMuPDF (`fitz`) to access PDF structure at the block and span level
- Extracts text with positional coordinates (x, y coordinates for each text element)
- Preserves indentation by calculating relative positioning from leftmost text
- Groups text lines into paragraphs based on vertical spacing thresholds
- Maintains reading order through vertical position sorting

**Key Features**:
- **Spatial Awareness**: Tracks x/y coordinates of all text elements
- **Indentation Preservation**: Calculates indent levels based on horizontal positioning (10 points = 1 indent level)
- **Paragraph Detection**: Groups lines with similar vertical positions as paragraphs
- **Layout Reconstruction**: Rebuilds formatted text that mirrors original document structure

**Output**: Clean, formatted text that preserves the visual hierarchy and structure of the original PDF page.

**Code Location**: `_extract_text_with_layout()` method

---

### Stage 2: Page Classification

**Purpose**: Automatically categorize each PDF page into one of four predefined types to understand document structure.

**Classification Categories**:
1. **Cover Page**: Title pages with author, publication information
2. **Table of Contents Page**: Pages listing chapters/sections with page numbers
3. **Chapter Beginning Page**: Pages that start new chapters or major sections
4. **Normal Text**: Regular content pages with paragraphs

**Technical Implementation**:
- Uses Ollama LLM API for intelligent classification
- Sends first 500 characters of page text to language model
- Implements retry mechanism (default: 3 attempts) for API reliability
- Includes conservative classification prompts to avoid false positives
- Validates responses against predefined categories

**Classification Process**:
1. Prepare system message with classification criteria
2. Send page text excerpt to Ollama model
3. Parse and validate LLM response
4. Map response to one of the four categories
5. Default to "Normal text" if classification fails

**Output**: Each page receives a classification label that helps identify document structure.

**Code Location**: `_classify_page()` method

---

### Stage 3: Conclusion/Future Work Detection

**Purpose**: Identify pages classified as "Chapter beginning" that specifically contain conclusion or future work content.

**Target Content Types**:
- **Conclusion Chapters**: "Conclusions", "Summary and Conclusions", "Final Remarks", "Discussion and Conclusions", "Concluding Remarks"
- **Future Work Chapters**: "Future Work", "Future Directions", "Future Research", "Recommendations", "Further Work", "Future Studies"

**Technical Implementation**:
- Only processes pages already classified as "Chapter beginning" in Stage 2
- Uses specialized LLM prompts focused on academic document structure
- Analyzes first 500 characters of chapter content for classification cues
- Distinguishes between main chapters and subsections (e.g., rejects "3.7 Conclusions" subsections)
- Returns binary classification (Yes/No) for conclusion/future work identification

**Classification Logic**:
1. Extract chapter content excerpt
2. Send to Ollama with specialized academic document prompts
3. LLM analyzes content for conclusion/future work indicators
4. Returns boolean result based on content analysis
5. Applies conservative criteria to avoid false positives

**Output**: Binary flag indicating whether a chapter beginning is specifically a conclusion or future work chapter.

**Code Location**: `_is_conclusion_chapter()` method

---

## Pipeline Integration

### Sequential Processing
Each PDF page flows through all three stages:
```
PDF Page → Stage 1 (Layout Extraction) → Stage 2 (Classification) → Stage 3 (if Chapter Beginning)
```

### Data Flow
1. **Stage 1** produces formatted text with preserved layout
2. **Stage 2** uses Stage 1 output to classify page type
3. **Stage 3** uses Stage 1 output for chapter beginnings identified in Stage 2

### Output Generation
- **Individual PDF Analysis**: JSON files with page-by-page results
- **Chapter Beginnings**: Consolidated text file with all identified chapter starts
- **Conclusion Chapters**: Separate file containing only conclusion/future work chapters
- **Progress Tracking**: JSON file enabling resumable processing

## Key Features

### Parallel Processing
- Supports multi-GPU processing with automatic load balancing
- Distributes PDFs across multiple Ollama instances
- Auto-detects available resources (GPUs, Ollama instances)
- Configurable worker count with intelligent defaults

### Robustness
- Retry mechanisms for API failures
- Progress tracking for resumable operations
- Comprehensive error handling and logging
- Conservative classification to minimize false positives

### Scalability
- Processes entire directory trees recursively
- Handles large document collections efficiently
- Timestamped output directories for experiment organization
- Memory-efficient processing of individual pages

## Usage

### Basic Usage
```bash
python enhanced_pdf_processor.py --directory /path/to/pdfs
```

### Parallel Processing
```bash
python enhanced_pdf_processor.py --directory /path/to/pdfs --parallel --workers 4
```

### Custom Configuration
```bash
python enhanced_pdf_processor.py \
  --directory /path/to/pdfs \
  --output /path/to/results \
  --model mistral-small \
  --parallel \
  --debug
```

## Requirements

- Python 3.7+
- PyMuPDF (`pip install pymupdf`)
- requests (`pip install requests`)
- tqdm (`pip install tqdm`)
- Running Ollama instance with compatible model

## Output Files

- `chapter_beginnings.txt`: All identified chapter beginnings
- `conclusion_beginnings.txt`: Conclusion and future work chapters only
- `{pdf_name}_analysis.json`: Individual PDF analysis results
- `pdf_processing_progress.json`: Progress tracking for resumable operations
- `pdf_processing.log`: Detailed processing logs

## Applications

This tool is particularly useful for:
- Academic research requiring systematic analysis of thesis/dissertation collections
- Literature review preparation
- Document structure analysis
- Content extraction for meta-analysis
- Educational research on academic writing patterns