#!/usr/bin/env python3
"""
Enhanced PDF Processing Pipeline

This script combines the functionality of multiple processors to create a streamlined,
single-pass pipeline that:
1. Processes PDFs with advanced layout preservation
2. Identifies chapter beginnings and conclusion sections simultaneously  
3. Tracks progress for resumable operations
4. Generates comprehensive output files

This replaces the need for separate directory_pdf_page_classifier.py, 
conclusion_extractor.py, and revised_conclusion_extractor.py scripts.
"""

import os
import re
import json
import time
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path
import fitz  # PyMuPDF
import requests
from tqdm import tqdm


def load_pdfs_from_csv(csv_path: str) -> List[str]:
    """
    Load PDF paths from ground truth CSV file.
    
    Args:
        csv_path: Path to CSV file with 'full_path' column
        
    Returns:
        List of existing PDF file paths from the CSV
    """
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        if 'full_path' not in df.columns:
            raise ValueError(f"CSV file {csv_path} must contain 'full_path' column")
        
        # Get unique paths
        unique_paths = df['full_path'].unique().tolist()
        
        # Filter to existing files only
        existing_paths = []
        missing_paths = []
        
        for path in unique_paths:
            if os.path.exists(path):
                existing_paths.append(path)
            else:
                missing_paths.append(path)
        
        print(f"Loaded {len(existing_paths)} existing PDFs from CSV: {csv_path}")
        if missing_paths:
            print(f"Warning: {len(missing_paths)} PDFs from CSV not found on filesystem")
        
        return existing_paths
        
    except Exception as e:
        print(f"Error loading PDFs from CSV {csv_path}: {e}")
        return []


class EnhancedPDFProcessor:
    """Enhanced PDF processor with integrated chapter and conclusion detection."""
    
    PAGE_CATEGORIES = ["Cover page", "Table of contents page", "Chapter beginning page", "Normal text"]
    
    def __init__(self, 
                 ollama_url: str = "http://localhost:11434/api/chat",
                 model: str = "mistral-small",
                 tracking_file: str = "pdf_processing_progress.json",
                 chapter_output_file: str = "chapter_beginnings.txt",
                 conclusion_output_file: str = "conclusion_beginnings.txt",
                 max_retries: int = 3,
                 request_timeout: int = 30,
                 max_concurrent_requests: int = 8,
                 debug: bool = False):
        """
        Initialize the Enhanced PDF Processor.
        
        Args:
            ollama_url: URL for Ollama API endpoint
            model: Ollama model to use for classification
            tracking_file: File to track processing progress
            chapter_output_file: File to save all chapter beginnings
            conclusion_output_file: File to save conclusion chapter beginnings
            max_retries: Number of retries for failed API calls
            request_timeout: Timeout for API requests in seconds
            max_concurrent_requests: Maximum number of concurrent Ollama requests
            debug: Enable debug output
        """
        self.ollama_url = ollama_url
        self.model = model
        self.tracking_file = tracking_file
        self.chapter_output_file = chapter_output_file
        self.conclusion_output_file = conclusion_output_file
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.max_concurrent_requests = max_concurrent_requests
        self.debug = debug
        
        # Data structures
        self.progress = self._load_progress()
        
        # Setup logging
        log_level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("pdf_processing.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_progress(self) -> Dict:
        """Load progress from tracking file if it exists."""
        if os.path.exists(self.tracking_file):
            try:
                with open(self.tracking_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                self.logger.warning("Error reading tracking file. Starting fresh.")
        
        return {"processed_files": [], "last_updated": time.time()}
    
    def _save_progress(self, pdf_path: str = None) -> None:
        """Save current progress to tracking file."""
        self.progress["last_updated"] = time.time()
        if pdf_path and pdf_path not in self.progress["processed_files"]:
            self.progress["processed_files"].append(pdf_path)
        
        with open(self.tracking_file, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, indent=2)
    
    def _save_chapter_beginning(self, pdf_path: str, page_num: int, page_text: str, is_conclusion: bool = False) -> None:
        """Save chapter beginning to the appropriate output file(s)."""
        # Always save to main chapters file
        with open(self.chapter_output_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'=' * 80}\n")
            f.write(f"PDF: {pdf_path} - Page: {page_num}\n")
            f.write(f"{'-' * 80}\n")
            f.write(page_text)
            f.write(f"\n{'=' * 80}\n")
        
        # If it's a conclusion, also save to conclusions file
        if is_conclusion:
            with open(self.conclusion_output_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'=' * 80}\n")
                f.write(f"PDF: {pdf_path} - Page: {page_num}\n")
                f.write(f"{'-' * 80}\n")
                f.write(page_text)
                f.write(f"\n{'=' * 80}\n")
    
    def parse_pdf(self, pdf_path: str) -> Dict:
        """
        Parse a PDF file with enhanced layout preservation and integrated classification.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dict containing parsed pages with their layouts preserved
        """
        # Progress tracking disabled - process all files
        # if pdf_path in self.progress["processed_files"]:
        #     self.logger.info(f"Skipping already processed file: {pdf_path}")
        #     return None
        
        self.logger.info(f"Parsing PDF: {pdf_path}")
        result = {
            "filename": os.path.basename(pdf_path),
            "pages": [],
            "chapter_pages": [],
            "conclusion_pages": []
        }
        
        # Open the PDF file
        try:
            with fitz.open(pdf_path) as doc:
                total_pages = len(doc)
                self.logger.info(f"Total pages: {total_pages}")
                
                # Process each page
                for page_num, page in enumerate(tqdm(doc, desc="Processing pages")):
                    # Extract text with enhanced layout preservation
                    page_text = self._extract_text_with_layout(page)
                    
                    # Classify the page
                    category = self._classify_page(page_text)
                    
                    # Store the results
                    page_result = {
                        "page_number": page_num + 1,
                        "text_with_layout": page_text,
                        "category": category
                    }
                    result["pages"].append(page_result)
                    
                    # If this is a chapter beginning, process further
                    if "chapter" in category.lower():
                        self.logger.info(f"Found chapter beginning in {pdf_path} on page {page_num + 1}")
                        
                        # Check if it's a conclusion chapter
                        is_conclusion = self._is_conclusion_chapter(page_text)
                        
                        if is_conclusion:
                            self.logger.info(f"Identified as conclusion chapter")
                            result["conclusion_pages"].append(page_num + 1)
                        
                        result["chapter_pages"].append(page_num + 1)
                        
                        # Save to appropriate files
                        self._save_chapter_beginning(pdf_path, page_num + 1, page_text, is_conclusion)
                
                # Mark this file as processed
                self._save_progress(pdf_path)
                return result
                
        except Exception as e:
            self.logger.error(f"Error processing {pdf_path}: {e}")
            return None
    
    def _extract_text_with_layout(self, page: fitz.Page) -> str:
        """
        Extract text from a page while preserving its layout using advanced positioning.
        
        This method uses the enhanced approach from improved_pdf_layout_parser.py
        """
        # Get page dimensions
        page_width, page_height = page.rect.width, page.rect.height
        
        # Extract text with dict mode to get positions
        blocks_dict = page.get_text("dict")
        
        # Create a list to store text lines with position info
        text_lines = []
        
        # Process each block
        for block in blocks_dict["blocks"]:
            if block["type"] != 0:  # Skip non-text blocks
                continue
                
            # Process each line
            for line in block["lines"]:
                line_text = ""
                line_x0 = float('inf')  # Track leftmost position for indentation
                
                # Process spans in the line
                for span in line["spans"]:
                    line_text += span["text"]
                    
                    # Track leftmost position
                    if span["bbox"][0] < line_x0:
                        line_x0 = span["bbox"][0]
                
                # Store line with position info
                if line_text.strip():  # Skip empty lines
                    text_lines.append({
                        "text": line_text,
                        "x0": line_x0,
                        "y0": line["bbox"][1],  # Use y0 for vertical sorting
                        "bbox": line["bbox"]
                    })
        
        # Sort lines by vertical position
        text_lines.sort(key=lambda x: x["y0"])
        
        # Group lines that are close vertically as paragraphs
        paragraphs = []
        current_paragraph = []
        last_y = None
        
        for line in text_lines:
            if last_y is None or abs(line["y0"] - last_y) > 10:  # Threshold for new paragraph
                if current_paragraph:
                    paragraphs.append(current_paragraph)
                current_paragraph = [line]
            else:
                current_paragraph.append(line)
            last_y = line["y0"]
        
        if current_paragraph:
            paragraphs.append(current_paragraph)
        
        # Find the minimum x position for indentation reference
        min_x = min([line["x0"] for line in text_lines]) if text_lines else 0
        
        # Build the formatted text
        formatted_text = ""
        
        for paragraph in paragraphs:
            for line in paragraph:
                # Calculate indentation
                indent_level = int((line["x0"] - min_x) / 10)  # 10 points = 1 indent level
                indent_spaces = " " * indent_level
                
                # Add the indented line
                formatted_text += indent_spaces + line["text"] + "\n"
            
            # Add extra newline between paragraphs
            formatted_text += "\n"
        
        # Remove excessive newlines
        formatted_text = re.sub(r'\n{3,}', '\n\n', formatted_text)
        
        return formatted_text
    
    def _classify_page(self, page_text: str) -> str:
        """
        Classify a page using Ollama with retry mechanism.
        """
        # Prepare system and user messages for Ollama
        system_message = """You are a PDF page classifier. Analyze the page text and classify it into exactly one of these categories: 
'Cover page', 'Table of contents page', 'Chapter beginning page', or 'Normal text'.

A cover page typically has the title, author, and publication information. 
A table of contents page lists chapters or sections with page numbers. 
A chapter beginning page usually starts with a chapter number or title and may have a drop cap or decorative element.
Normal text pages contain regular paragraphs of content.

Respond with ONLY the category name, nothing else."""
        
        user_message = f"""Analyze this page text and classify it:
```
Be conservative - only mark as chapter beginning if you're confident this page starts a new chapter, not just a section within a chapter.

{page_text[:500]}
```"""

        # Call Ollama API with retries
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.ollama_url,
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": user_message}
                        ],
                        "stream": False,
                        "temperature": 0
                    },
                    timeout=self.request_timeout
                )
                response.raise_for_status()
                result = response.json().get("message", {}).get("content", "").strip()
                
                # Validate and normalize the result
                for category in self.PAGE_CATEGORIES:
                    if category.lower() in result.lower():
                        return category
                
                # Default to "Normal text" if no clear match
                return "Normal text"
                
            except Exception as e:
                self.logger.warning(f"Error calling Ollama (attempt {attempt+1}): {e}")
                if attempt == self.max_retries - 1:
                    self.logger.error("All Ollama attempts failed, defaulting to 'Normal text'")
                    return "Normal text"
        
        return "Normal text"
    
    def _is_conclusion_chapter(self, chapter_content: str) -> bool:
        """
        Determine if a chapter is relevant (conclusion, future work, summary, discussion, recommendations, limitations, or implications) using binary LLM classification.
        """
        system_message = """You are a research paper analyzer. Your task is to determine if a given text is the beginning of a MAIN chapter that covers conclusions, future work, or related content from an academic paper or thesis.

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

Respond with ONLY 'Yes' or 'No'."""
        
        # Take first part of chapter content for classification
        content_excerpt = chapter_content[:500]
        
        user_message = f"""Analyze this text and determine if it's the beginning of a main chapter covering conclusions, future work, summaries, discussions, recommendations, limitations, or implications:

```
{content_excerpt}
```

Is this the beginning of a main chapter in one of these categories?

Respond with ONLY 'Yes' or 'No'."""

        # Call Ollama API with retries
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.ollama_url,
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": user_message}
                        ],
                        "stream": False,
                        "temperature": 0
                    },
                    timeout=self.request_timeout
                )
                response.raise_for_status()
                result = response.json().get("message", {}).get("content", "").strip()
                
                # Check if the result contains "yes" (case insensitive)
                is_conclusion = "yes" in result.lower().strip()
                
                if self.debug:
                    self.logger.debug(f"Binary classification result: {result} -> {is_conclusion}")
                
                return is_conclusion
                
            except Exception as e:
                self.logger.warning(f"Error classifying conclusion (attempt {attempt+1}): {e}")
                if attempt == self.max_retries - 1:
                    self.logger.error("All conclusion classification attempts failed, defaulting to False")
                    return False
        
        return False
    
    def _create_timestamped_output_dir(self, base_dir: str) -> str:
        """
        Create a timestamped output directory for this processing run.
        
        Args:
            base_dir: Base directory where the timestamped folder should be created
            
        Returns:
            Path to the timestamped output directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.model.replace("-", "_")  # Replace hyphens for folder names
        timestamped_dir = os.path.join(base_dir, f"results_{timestamp}_{model_name}")
        
        if not os.path.exists(timestamped_dir):
            os.makedirs(timestamped_dir)
            self.logger.info(f"Created timestamped output directory: {timestamped_dir}")
        
        return timestamped_dir
    
    def _detect_ollama_instances(self) -> List[str]:
        """
        Detect running Ollama instances by checking common ports.
        
        Returns:
            List of available Ollama URLs
        """
        common_ports = [11434, 11435, 11436, 11437, 11438, 11439]  # Check common Ollama ports
        available_instances = []
        
        for port in common_ports:
            url = f"http://localhost:{port}"
            try:
                response = requests.get(f"{url}/api/version", timeout=2)
                if response.status_code == 200:
                    available_instances.append(f"{url}/api/chat")
                    self.logger.debug(f"Found Ollama instance at {url}")
            except requests.RequestException:
                continue
        
        if not available_instances:
            # Fallback to default
            available_instances = [self.ollama_url]
            self.logger.warning("No additional Ollama instances detected, using default URL")
        
        return available_instances
    
    def _detect_available_gpus(self) -> int:
        """
        Detect number of available GPUs.
        
        Returns:
            Number of available GPUs
        """
        try:
            # Check CUDA_VISIBLE_DEVICES first
            cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
            if cuda_devices:
                gpu_count = len([d for d in cuda_devices.split(',') if d.strip()])
                self.logger.debug(f"Found {gpu_count} GPUs from CUDA_VISIBLE_DEVICES")
                return gpu_count
            
            # Fallback to nvidia-smi
            result = os.popen('nvidia-smi --list-gpus 2>/dev/null').read()
            gpu_count = len(result.strip().split('\n')) if result.strip() else 1
            self.logger.debug(f"Found {gpu_count} GPUs from nvidia-smi")
            return max(1, gpu_count)
            
        except Exception:
            self.logger.warning("Could not detect GPU count, defaulting to 1")
            return 1
    
    def save_results(self, results: Dict, output_path: str) -> None:
        """
        Save parsing and classification results to a JSON file.
        """
        if results:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Results saved to {output_path}")
    
    def process_directory(self, directory_path: str, output_dir: str = None, use_timestamped_dir: bool = True) -> None:
        """
        Process all PDF files in a directory and its subdirectories with enhanced processing.
        
        Args:
            directory_path: Path to the directory to process
            output_dir: Directory to save individual PDF analysis results (if None, use directory_path)
            use_timestamped_dir: Whether to create a timestamped subdirectory for results
        """
        if output_dir is None:
            output_dir = directory_path
        
        # Create timestamped output directory if requested
        if use_timestamped_dir:
            output_dir = self._create_timestamped_output_dir(output_dir)
            # Update chapter and conclusion output files to be in the timestamped directory
            self.chapter_output_file = os.path.join(output_dir, os.path.basename(self.chapter_output_file))
            self.conclusion_output_file = os.path.join(output_dir, os.path.basename(self.conclusion_output_file))
            # Update progress file to be in the timestamped directory for experiment-specific tracking
            self.tracking_file = os.path.join(output_dir, os.path.basename(self.tracking_file))
            # Reload progress from the new location
            self.progress = self._load_progress()
        elif not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Find all PDF files in the directory and subdirectories
        pdf_files = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        
        self.logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Initialize counters
        total_chapters = 0
        total_conclusions = 0
        processed_count = 0
        
        # Process each PDF file
        for pdf_path in pdf_files:
            # Progress tracking disabled - process all files
            # if pdf_path in self.progress["processed_files"]:
            #     self.logger.info(f"Skipping already processed: {pdf_path}")
            #     continue
            
            # Generate output path for this PDF's results
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_analysis.json")
            
            # Process the PDF
            try:
                results = self.parse_pdf(pdf_path)
                if results:
                    self.save_results(results, output_path)
                    total_chapters += len(results.get("chapter_pages", []))
                    total_conclusions += len(results.get("conclusion_pages", []))
                    processed_count += 1
            except Exception as e:
                self.logger.error(f"Error processing {pdf_path}: {e}")
            
            # Save progress after each file
            self._save_progress(pdf_path)
        
        # Print summary
        self.logger.info(f"\nProcessing Complete!")
        self.logger.info(f"Processed PDFs: {processed_count}")
        self.logger.info(f"Total chapter beginnings found: {total_chapters}")
        self.logger.info(f"Total conclusion chapters found: {total_conclusions}")
        self.logger.info(f"All chapters saved to: {self.chapter_output_file}")
        self.logger.info(f"Conclusion chapters saved to: {self.conclusion_output_file}")
    
    def process_pdf_list(self, pdf_files: List[str], output_dir: str = None, use_timestamped_dir: bool = True) -> None:
        """
        Process a specific list of PDF files with enhanced processing.
        
        Args:
            pdf_files: List of PDF file paths to process
            output_dir: Directory to save individual PDF analysis results (if None, use current directory)
            use_timestamped_dir: Whether to create a timestamped subdirectory for results
        """
        if output_dir is None:
            output_dir = os.getcwd()
        
        # Create timestamped output directory if requested
        if use_timestamped_dir:
            output_dir = self._create_timestamped_output_dir(output_dir)
            # Update chapter and conclusion output files to be in the timestamped directory
            self.chapter_output_file = os.path.join(output_dir, os.path.basename(self.chapter_output_file))
            self.conclusion_output_file = os.path.join(output_dir, os.path.basename(self.conclusion_output_file))
            # Update progress file to be in the timestamped directory for experiment-specific tracking
            self.tracking_file = os.path.join(output_dir, os.path.basename(self.tracking_file))
            # Reload progress from the new location
            self.progress = self._load_progress()
        elif not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.logger.info(f"Processing {len(pdf_files)} PDF files from CSV")
        
        # Initialize counters
        total_chapters = 0
        total_conclusions = 0
        processed_count = 0
        
        # Process each PDF file
        for pdf_path in pdf_files:
            # Progress tracking disabled - process all files
            # if pdf_path in self.progress["processed_files"]:
            #     self.logger.info(f"Skipping already processed: {pdf_path}")
            #     continue
            
            # Process the PDF
            result = self.parse_pdf(pdf_path)
            if result:
                # Save individual results to JSON file
                output_filename = f"{Path(pdf_path).stem}_analysis.json"
                output_filepath = os.path.join(output_dir, output_filename)
                
                try:
                    with open(output_filepath, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    self.logger.error(f"Error saving results for {pdf_path}: {e}")
                
                total_chapters += len(result.get("chapter_pages", []))
                total_conclusions += len(result.get("conclusion_pages", []))
                processed_count += 1
        
        self.logger.info(f"Processing complete!")
        self.logger.info(f"Processed: {processed_count} PDFs")
        self.logger.info(f"Total chapters found: {total_chapters}")
        self.logger.info(f"Total conclusions found: {total_conclusions}")
        self.logger.info(f"All chapters saved to: {self.chapter_output_file}")
        self.logger.info(f"Conclusion chapters saved to: {self.conclusion_output_file}")
    
    def process_pdf_list_parallel(self, pdf_files: List[str], output_dir: str = None, num_workers: int = None, use_timestamped_dir: bool = True) -> None:
        """
        Process a specific list of PDF files in parallel with enhanced processing.
        
        Args:
            pdf_files: List of PDF file paths to process
            output_dir: Directory to save individual PDF analysis results (if None, use current directory)
            num_workers: Number of parallel workers (if None, uses CPU count)
            use_timestamped_dir: Whether to create a timestamped subdirectory for results
        """
        if output_dir is None:
            output_dir = os.getcwd()
        
        # Create timestamped output directory if requested
        if use_timestamped_dir:
            output_dir = self._create_timestamped_output_dir(output_dir)
            # Update chapter and conclusion output files to be in the timestamped directory
            self.chapter_output_file = os.path.join(output_dir, os.path.basename(self.chapter_output_file))
            self.conclusion_output_file = os.path.join(output_dir, os.path.basename(self.conclusion_output_file))
            # Update progress file to be in the timestamped directory for experiment-specific tracking
            self.tracking_file = os.path.join(output_dir, os.path.basename(self.tracking_file))
            # Reload progress from the new location
            self.progress = self._load_progress()
        elif not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Auto-detect available resources
        available_ollama_instances = self._detect_ollama_instances()
        num_gpus = self._detect_available_gpus()
        
        # Auto-detect number of workers based on available resources
        if num_workers is None:
            num_workers = min(len(available_ollama_instances), num_gpus, 6)
            self.logger.info(f"Auto-detected {num_gpus} GPUs and {len(available_ollama_instances)} Ollama instances, using {num_workers} workers")
        
        if not available_ollama_instances:
            self.logger.error("No Ollama instances found. Falling back to single-threaded processing.")
            self.process_pdf_list(pdf_files, output_dir, use_timestamped_dir=False)
            return
        
        self.logger.info(f"Found {len(available_ollama_instances)} Ollama instances for parallel processing")
        self.logger.info(f"Using {num_workers} workers to process {len(pdf_files)} PDFs")
        
        # Progress tracking disabled - process all files
        # unprocessed_files = [f for f in pdf_files if f not in self.progress["processed_files"]]
        unprocessed_files = pdf_files
        
        if not unprocessed_files:
            self.logger.info("No files to process!")
            return
        
        # Create processor configuration for workers
        processor_config = {
            'available_ollama_instances': available_ollama_instances,
            'model': self.model,
            'tracking_file': self.tracking_file,  # Now points to experiment-specific progress file
            'chapter_output_file': self.chapter_output_file,
            'conclusion_output_file': self.conclusion_output_file,
            'max_retries': self.max_retries,
            'request_timeout': self.request_timeout,
            'debug': self.debug
        }
        
        # Prepare arguments for workers
        worker_args = [(pdf_path, output_dir, processor_config) for pdf_path in unprocessed_files]
        
        # Initialize counters
        total_chapters = 0
        total_conclusions = 0
        processed_count = 0
        
        # Process files in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_pdf = {executor.submit(process_single_pdf_worker, args): args[0] 
                           for args in worker_args}
            
            # Process completed tasks with progress bar
            with tqdm(total=len(unprocessed_files), desc="Processing PDFs") as pbar:
                for future in as_completed(future_to_pdf):
                    pdf_path = future_to_pdf[future]
                    try:
                        pdf_path, results, success = future.result()
                        if success and results:
                            total_chapters += len(results.get("chapter_pages", []))
                            total_conclusions += len(results.get("conclusion_pages", []))
                            processed_count += 1
                        
                        pbar.set_postfix({
                            'processed': processed_count,
                            'chapters': total_chapters,
                            'conclusions': total_conclusions
                        })
                        
                    except Exception as e:
                        self.logger.error(f"Error processing {pdf_path}: {e}")
                    
                    pbar.update(1)
        
        self.logger.info(f"Parallel processing complete!")
        self.logger.info(f"Processed: {processed_count} PDFs")
        self.logger.info(f"Total chapters found: {total_chapters}")
        self.logger.info(f"Total conclusions found: {total_conclusions}")
        self.logger.info(f"All chapters saved to: {self.chapter_output_file}")
        self.logger.info(f"Conclusion chapters saved to: {self.conclusion_output_file}")
    
    def process_directory_parallel(self, directory_path: str, output_dir: str = None, num_workers: int = None, use_timestamped_dir: bool = True) -> None:
        """
        Process all PDF files in a directory using parallel processing to utilize multiple GPUs.
        
        Args:
            directory_path: Path to the directory to process
            output_dir: Directory to save individual PDF analysis results (if None, use directory_path)
            num_workers: Number of parallel workers (auto-detects based on available GPUs if None)
            use_timestamped_dir: Whether to create a timestamped subdirectory for results
        """
        if output_dir is None:
            output_dir = directory_path
        
        # Create timestamped output directory if requested
        if use_timestamped_dir:
            output_dir = self._create_timestamped_output_dir(output_dir)
            # Update chapter and conclusion output files to be in the timestamped directory
            self.chapter_output_file = os.path.join(output_dir, os.path.basename(self.chapter_output_file))
            self.conclusion_output_file = os.path.join(output_dir, os.path.basename(self.conclusion_output_file))
            # Update progress file to be in the timestamped directory for experiment-specific tracking
            self.tracking_file = os.path.join(output_dir, os.path.basename(self.tracking_file))
            # Reload progress from the new location
            self.progress = self._load_progress()
        elif not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Auto-detect available resources
        available_ollama_instances = self._detect_ollama_instances()
        num_gpus = self._detect_available_gpus()
        
        # Auto-detect number of workers based on available resources
        if num_workers is None:
            num_workers = min(len(available_ollama_instances), num_gpus, 4)  # Cap at 4 to avoid overwhelming
            self.logger.info(f"Auto-detected {num_gpus} GPUs and {len(available_ollama_instances)} Ollama instances, using {num_workers} workers")
        
        # Find all PDF files in the directory and subdirectories
        pdf_files = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        
        # Progress tracking disabled - process all files
        # unprocessed_files = [pdf for pdf in pdf_files if pdf not in self.progress["processed_files"]]
        unprocessed_files = pdf_files
        
        self.logger.info(f"Found {len(pdf_files)} PDF files total")
        self.logger.info(f"Processing {len(unprocessed_files)} unprocessed files with {num_workers} workers")
        
        if not unprocessed_files:
            self.logger.info("All files already processed!")
            return
        
        # Create processor configuration for workers
        processor_config = {
            'available_ollama_instances': available_ollama_instances,
            'model': self.model,
            'tracking_file': self.tracking_file,  # Now points to experiment-specific progress file
            'chapter_output_file': self.chapter_output_file,
            'conclusion_output_file': self.conclusion_output_file,
            'max_retries': self.max_retries,
            'request_timeout': self.request_timeout,
            'debug': self.debug
        }
        
        # Prepare arguments for workers
        worker_args = [(pdf_path, output_dir, processor_config) for pdf_path in unprocessed_files]
        
        # Initialize counters
        total_chapters = 0
        total_conclusions = 0
        processed_count = 0
        
        # Process files in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_pdf = {executor.submit(process_single_pdf_worker, args): args[0] 
                           for args in worker_args}
            
            # Process completed tasks with progress bar
            with tqdm(total=len(unprocessed_files), desc="Processing PDFs") as pbar:
                for future in as_completed(future_to_pdf):
                    pdf_path = future_to_pdf[future]
                    try:
                        pdf_path, results, success = future.result()
                        if success and results:
                            total_chapters += len(results.get("chapter_pages", []))
                            total_conclusions += len(results.get("conclusion_pages", []))
                            processed_count += 1
                        
                        pbar.set_postfix({
                            'processed': processed_count,
                            'chapters': total_chapters,
                            'conclusions': total_conclusions
                        })
                        
                    except Exception as e:
                        self.logger.error(f"Error processing {pdf_path}: {e}")
                    
                    pbar.update(1)
        
        # Reload progress to get the latest state from all workers
        self.progress = self._load_progress()
        
        # Print summary
        self.logger.info(f"\nParallel Processing Complete!")
        self.logger.info(f"Processed PDFs: {processed_count}")
        self.logger.info(f"Total chapter beginnings found: {total_chapters}")
        self.logger.info(f"Total conclusion chapters found: {total_conclusions}")
        self.logger.info(f"All chapters saved to: {self.chapter_output_file}")
        self.logger.info(f"Conclusion chapters saved to: {self.conclusion_output_file}")


def process_single_pdf_worker(args):
    """
    Worker function for parallel PDF processing with load balancing across Ollama instances.
    
    Args:
        args: Tuple containing (pdf_path, output_dir, processor_config)
    
    Returns:
        Tuple of (pdf_path, results, success_flag)
    """
    pdf_path, output_dir, processor_config = args
    
    # Select Ollama instance for this worker (round-robin based on process ID)
    available_instances = processor_config['available_ollama_instances']
    worker_id = os.getpid() % len(available_instances)
    selected_ollama_url = available_instances[worker_id]
    
    # Create a new processor instance for this worker
    processor = EnhancedPDFProcessor(
        ollama_url=selected_ollama_url,
        model=processor_config['model'],
        tracking_file=processor_config['tracking_file'],
        chapter_output_file=processor_config['chapter_output_file'],
        conclusion_output_file=processor_config['conclusion_output_file'],
        max_retries=processor_config['max_retries'],
        request_timeout=processor_config['request_timeout'],
        debug=processor_config['debug']
    )
    
    try:
        # Generate output path for this PDF's results
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_analysis.json")
        
        # Log which Ollama instance is being used for this worker
        if processor_config.get('debug'):
            print(f"Worker processing {pdf_path} using Ollama instance: {selected_ollama_url}")
        
        # Process the PDF
        results = processor.parse_pdf(pdf_path)
        if results:
            processor.save_results(results, output_path)
            # Save progress after processing
            processor._save_progress(pdf_path)
            return pdf_path, results, True
        else:
            return pdf_path, None, False
            
    except Exception as e:
        # Use print instead of logger to avoid conflicts in multiprocessing
        print(f"Error processing {pdf_path} with {selected_ollama_url}: {e}")
        return pdf_path, None, False


def main():
    """Main function to run the Enhanced PDF Processor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced PDF processing with integrated chapter and conclusion detection")
    parser.add_argument("--csv_path", default="../final_groundtruth_filtered.csv", 
                        help="CSV file with ground truth containing full_path column")
    parser.add_argument("--directory", "-d", 
                        help="Optional: Directory to recursively search for PDF files (legacy mode)")
    parser.add_argument("--output", "-o", default=None, 
                        help="Output directory for analysis results (defaults to input directory)")
    parser.add_argument("--tracking", "-t", default="pdf_processing_progress.json", 
                        help="File to track processing progress")
    parser.add_argument("--chapters", "-c", default="chapter_beginnings.txt", 
                        help="File to save all chapter beginnings")
    parser.add_argument("--conclusions", default="conclusion_beginnings.txt", 
                        help="File to save conclusion chapter beginnings")
    parser.add_argument("--ollama-url", default="http://localhost:11434/api/chat", 
                        help="Ollama API endpoint")
    parser.add_argument("--model", default="mistral-small", 
                        help="Ollama model to use")
    parser.add_argument("--retries", type=int, default=3,
                        help="Max retries for API calls")
    parser.add_argument("--timeout", type=int, default=30,
                        help="API timeout in seconds")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output")
    parser.add_argument("--parallel", action="store_true",
                        help="Use parallel processing to utilize multiple GPUs")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (auto-detects based on GPUs if not specified)")
    parser.add_argument("--no-timestamp", action="store_true",
                        help="Disable timestamped output directories (save directly to output directory)")
    parser.add_argument("--max-concurrent", type=int, default=8,
                        help="Maximum number of concurrent Ollama requests (default: 8)")
    
    args = parser.parse_args()
    
    processor = EnhancedPDFProcessor(
        ollama_url=args.ollama_url,
        model=args.model,
        tracking_file=args.tracking,
        chapter_output_file=args.chapters,
        conclusion_output_file=args.conclusions,
        max_retries=args.retries,
        request_timeout=args.timeout,
        max_concurrent_requests=args.max_concurrent,
        debug=args.debug
    )
    
    # Ensure output files are saved in the correct output directory
    if args.output:
        processor.chapter_output_file = os.path.join(args.output, os.path.basename(processor.chapter_output_file))
        processor.conclusion_output_file = os.path.join(args.output, os.path.basename(processor.conclusion_output_file))
        processor.tracking_file = os.path.join(args.output, os.path.basename(processor.tracking_file))
    
    # Determine PDF files to process
    if args.csv_path and os.path.exists(args.csv_path):
        # CSV-based loading (primary mode)
        pdf_files = load_pdfs_from_csv(args.csv_path)
        if not pdf_files:
            print(f"No valid PDFs found in CSV: {args.csv_path}")
            return
        
        # Use CSV mode processing
        use_timestamps = not args.no_timestamp
        if args.parallel:
            processor.process_pdf_list_parallel(pdf_files, args.output, args.workers, use_timestamps)
        else:
            processor.process_pdf_list(pdf_files, args.output, use_timestamps)
            
    elif args.directory:
        # Legacy directory mode
        if not os.path.exists(args.directory):
            print(f"Directory does not exist: {args.directory}")
            return
        
        use_timestamps = not args.no_timestamp
        if args.parallel:
            processor.process_directory_parallel(args.directory, args.output, args.workers, use_timestamps)
        else:
            processor.process_directory(args.directory, args.output, use_timestamps)
    else:
        print("Error: Must provide either --csv_path (default) or --directory")
        return


if __name__ == "__main__":
    main()