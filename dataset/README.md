# Dataset Metrics Analysis

This script analyzes dissertation PDFs from the VT ETD dataset and provides comprehensive metrics including PDF counts, page statistics, and discipline distribution.

## Features

- **Parallel Processing**: Uses multiprocessing to analyze datasets efficiently
- **Progress Tracking**: tqdm progress bars show real-time processing status  
- **Flexible Analysis**: Can analyze large dataset, subset, or both
- **Department Extraction**: Automatically extracts department information from XML metadata
- **Page Count Analysis**: Gets min, max, and average page counts from PDFs
- **JSON Export**: Saves detailed results to JSON file

## Requirements

```bash
pip install tqdm
```

The script also requires `pdfinfo` (usually part of `poppler-utils`):
```bash
sudo apt-get install poppler-utils  # Ubuntu/Debian
```

## Usage

### Basic Commands

1. **Analyze subset only** (fastest, good for testing):
```bash
python analyze_dataset_metrics.py --subset-only
```

2. **Analyze large dataset only** (13,071 directories):
```bash
python analyze_dataset_metrics.py --large-only
```

3. **Analyze both datasets** (comprehensive analysis):
```bash
python analyze_dataset_metrics.py
```

### Advanced Options

- **Control worker processes**:
```bash
python analyze_dataset_metrics.py --workers 16  # Use 16 CPU cores
```

- **Help**:
```bash
python analyze_dataset_metrics.py --help
```

## Expected Performance

- **Subset analysis**: ~302 PDFs, completes in seconds
- **Large dataset**: ~13,071 directories, estimated time depends on:
  - Number of CPU cores used
  - Actual number of PDFs found
  - PDF file sizes (for page count extraction)

With 16 cores, expect the large dataset to take several minutes to hours depending on the total number of PDFs.

## Output

The script provides:

1. **Console output**: Real-time progress and summary statistics
2. **JSON report**: Detailed metrics saved to `dataset_metrics_report.json`

### Sample Output

```
SUBSET DATASET METRICS:
----------------------------------------
Total PDFs: 302
Min pages: 41  
Max pages: 637
Average pages: 198.12
Total pages: 59,833
PDFs with page info: 302

Department Distribution:
  Electrical and Computer Engineering: 30
  Mechanical Engineering: 21
  Teaching and Learning: 20
  ...
```

## Dataset Structure

- **Large dataset**: `/projects/open_etds/etd/dissertation/`
  - Contains 13,071+ subdirectories
  - Each directory has XML files with metadata and PDF files
  - Department info extracted from `dublin_core.xml`

- **Subset dataset**: `final_groundtruth_filtered.csv`
  - Contains 302 unique PDFs
  - Department info already in CSV columns
  - Represents filtered/curated subset of larger dataset

## Notes

- The script handles missing files gracefully
- Progress bars show processing speed in real-time
- Memory usage is optimized through streaming processing
- Failed PDF page extractions are logged but don't stop processing