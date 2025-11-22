#!/usr/bin/env python3

"""
Multi-GPU header detection script that processes all PDFs from final_groundtruth.csv
using the same methodology as experiment.py, with each worker utilizing one GPU.
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

import fitz  # PyMuPDF
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm

from transformers import (
    AutoProcessor,
    AutoModelForTokenClassification,
)


def softmax(logits: np.ndarray, axis=-1):
    """Simple softmax implementation"""
    e = np.exp(logits - np.max(logits, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

def get_page_image(page, dpi=144):
    """Convert PDF page to PIL Image"""
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img

def normalize_box(bbox, page_width, page_height):
    """Normalize bounding box to 0-1000 scale for LayoutLMv3"""
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(1000, int(1000 * x0 / page_width)))
    y0 = max(0, min(1000, int(1000 * y0 / page_height)))
    x1 = max(0, min(1000, int(1000 * x1 / page_width)))
    y1 = max(0, min(1000, int(1000 * y1 / page_height)))
    
    # Ensure minimum dimensions
    x1 = max(x1, x0 + 1)
    y1 = max(y1, y0 + 1)
    
    return [x0, y0, x1, y1]

def group_nearby_headers(headers, line_tolerance=20, word_gap_tolerance=50, multiline=False):
    """
    Group nearby header words into complete header titles.
    
    Args:
        headers: List of detected header words with bbox and confidence
        line_tolerance: Vertical distance to consider words on same line (pixels)
        word_gap_tolerance: Horizontal gap to consider words part of same header (pixels)
        multiline: If True, allow grouping across lines (ignore horizontal gaps)
    
    Returns:
        List of grouped headers with combined text and bbox
    """
    if not headers:
        return []
    
    # Sort headers by reading order (top to bottom, left to right)
    sorted_headers = sorted(headers, key=lambda h: (h['bbox'][1], h['bbox'][0]))
    
    grouped_headers = []
    current_group = []
    
    for i, header in enumerate(sorted_headers):
        if not current_group:
            # Start new group
            current_group = [header]
        else:
            # Check if this header should be grouped with ANY header in current group
            should_group = False
            best_match = None
            min_distance = float('inf')
            
            for group_header in current_group:
                # Check vertical alignment (same line)
                y_center = (header['bbox'][1] + header['bbox'][3]) / 2
                group_y_center = (group_header['bbox'][1] + group_header['bbox'][3]) / 2
                vertical_distance = abs(y_center - group_y_center)
                
                # Check horizontal relationship
                # Case 1: header is to the right of group_header
                if header['bbox'][0] >= group_header['bbox'][2]:
                    horizontal_gap = header['bbox'][0] - group_header['bbox'][2]
                # Case 2: header is to the left of group_header  
                elif header['bbox'][2] <= group_header['bbox'][0]:
                    horizontal_gap = group_header['bbox'][0] - header['bbox'][2]
                # Case 3: headers overlap horizontally
                else:
                    horizontal_gap = 0
                
                # Check if they should be grouped
                if multiline:
                    # For multiline headers, only check vertical distance
                    if vertical_distance <= line_tolerance:
                        total_distance = vertical_distance
                        if total_distance < min_distance:
                            should_group = True
                            best_match = group_header
                            min_distance = total_distance
                else:
                    # Standard grouping: check both vertical and horizontal constraints
                    if (vertical_distance <= line_tolerance and 
                        horizontal_gap <= word_gap_tolerance):
                        total_distance = vertical_distance + horizontal_gap * 0.1  # Prefer closer matches
                        if total_distance < min_distance:
                            should_group = True
                            best_match = group_header
                            min_distance = total_distance
            
            if should_group:
                # Add to current group
                current_group.append(header)
            else:
                # Finalize current group and start new one
                if current_group:
                    merged = _merge_group(current_group, multiline)
                    grouped_headers.append(merged)
                current_group = [header]
    
    # Don't forget the last group
    if current_group:
        merged = _merge_group(current_group, multiline)
        grouped_headers.append(merged)
    
    return grouped_headers

def _merge_group(group, multiline=False):
    """Merge a group of header words into a single header"""
    if len(group) == 1:
        return {
            "text": group[0]["text"],
            "bbox": group[0]["bbox"],
            "confidence": group[0]["confidence"],
            "word_count": 1,
            "word_indices": [group[0]["word_index"]]
        }
    
    # Choose sorting strategy based on grouping type
    if multiline:
        # For multiline: preserve reading order (top-to-bottom, then left-to-right)
        group = sorted(group, key=lambda h: (h['bbox'][1], h['bbox'][0]))
    else:
        # For single-line: sort by horizontal position only
        group = sorted(group, key=lambda h: h['bbox'][0])
    
    # Combine text with spaces
    combined_text = " ".join(h["text"] for h in group)
    
    # Calculate bounding box that encompasses all words
    min_x = min(h["bbox"][0] for h in group)
    min_y = min(h["bbox"][1] for h in group)
    max_x = max(h["bbox"][2] for h in group)
    max_y = max(h["bbox"][3] for h in group)
    combined_bbox = [min_x, min_y, max_x, max_y]
    
    # Average confidence (weighted by word length)
    total_chars = sum(len(h["text"]) for h in group)
    if total_chars > 0:
        avg_confidence = sum(h["confidence"] * len(h["text"]) for h in group) / total_chars
    else:
        avg_confidence = sum(h["confidence"] for h in group) / len(group)
    
    return {
        "text": combined_text,
        "bbox": combined_bbox,
        "confidence": avg_confidence,
        "word_count": len(group),
        "word_indices": [h["word_index"] for h in group]
    }

class SectionHeaderDetector:
    def __init__(self, model_name="Mit1208/layoutlmv3-finetuned-DocLayNet", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model {model_name} on {self.device}")
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Get label mappings
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.label2id = {v: k for k, v in self.id2label.items()}
        
        # Find section header label
        self.section_header_id = self._find_section_header_label()
        if self.section_header_id is None:
            raise ValueError(f"Could not find section header label in {list(self.id2label.values())}")
        
        print(f"Using label ID {self.section_header_id}: '{self.id2label[self.section_header_id]}'")
    
    def _find_section_header_label(self):
        """Find the section header label ID"""
        candidates = [
            "section-header", "Section-header", "SECTION-HEADER", 
            "section_header", "Section_Header", "Header"
        ]
        
        for label_id, label_name in self.id2label.items():
            if any(candidate.lower() in label_name.lower() for candidate in ["section", "header"]):
                if "section" in label_name.lower() and "header" in label_name.lower():
                    return label_id
        
        return None
    
    @torch.no_grad()
    def detect_headers(self, page, confidence_threshold=0.5, max_tokens=512, 
                      group_headers=True, line_tolerance=20, word_gap_tolerance=50, multiline=False):
        """
        Detect section headers on a PDF page
        Returns list of detected headers with confidence scores
        """
        # Get page dimensions
        page_width = page.rect.width
        page_height = page.rect.height
        
        # Extract words with bounding boxes
        words_data = page.get_text("words")
        if not words_data:
            return []
        
        # Prepare inputs
        words = []
        boxes = []
        for word_info in words_data:
            if len(word_info) >= 5:
                x0, y0, x1, y1, text = word_info[:5]
                if text.strip():
                    words.append(text)
                    boxes.append(normalize_box([x0, y0, x1, y1], page_width, page_height))
        
        if not words:
            return []
        
        # Limit tokens to avoid memory issues
        if len(words) > max_tokens - 50:  # Leave room for special tokens
            words = words[:max_tokens - 50]
            boxes = boxes[:max_tokens - 50]
        
        # Get page image
        image = get_page_image(page)
        
        # Process with LayoutLMv3
        try:
            inputs = self.processor(
                image,
                words,
                boxes=boxes,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_tokens
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            outputs = self.model(**inputs)
            logits = outputs.logits[0].cpu().numpy()  # (seq_len, num_labels)
            probabilities = softmax(logits, axis=-1)
            
            # Extract section header probabilities
            header_probs = probabilities[:, self.section_header_id]
            
            # Map back to words (skip special tokens)
            detected_headers = []
            word_idx = 0
            
            for token_idx in range(len(header_probs)):
                # Skip special tokens and padding
                if token_idx < len(inputs['input_ids'][0]):
                    token_id = inputs['input_ids'][0][token_idx].item()
                    if token_id not in [self.processor.tokenizer.pad_token_id, 
                                      self.processor.tokenizer.cls_token_id, 
                                      self.processor.tokenizer.sep_token_id]:
                        
                        prob = float(header_probs[token_idx])
                        if prob >= confidence_threshold and word_idx < len(words):
                            detected_headers.append({
                                "text": words[word_idx],
                                "bbox": boxes[word_idx],
                                "confidence": prob,
                                "word_index": word_idx
                            })
                        
                        word_idx += 1
                        if word_idx >= len(words):
                            break
            
            # Group headers if requested
            if group_headers:
                grouped = group_nearby_headers(detected_headers, line_tolerance, word_gap_tolerance, multiline)
                return grouped
            else:
                return detected_headers
            
        except Exception as e:
            print(f"Error processing page: {e}")
            return []

def process_single_pdf(pdf_path: str, output_dir: str, confidence_threshold: float = 0.5, 
                       group_headers: bool = True, line_tolerance: int = 20, 
                       word_gap_tolerance: int = 50, multiline: bool = False, gpu_id: int = 0):
    """Process a single PDF and extract section headers using specified GPU"""
    
    # Set the GPU for this worker
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    
    # Set CUDA device if using GPU
    if torch.cuda.is_available() and gpu_id is not None:
        torch.cuda.set_device(gpu_id)
        # Clear GPU cache
        torch.cuda.empty_cache()
    
    try:
        # Initialize detector for this process
        detector = SectionHeaderDetector(device=device)
        
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if output already exists
        output_file = output_dir / f"{pdf_path.stem}.json"
        if output_file.exists():
            print(f"Skipping {pdf_path.name} - output already exists")
            return f"Skipped {pdf_path.name}"
        
        doc = fitz.open(pdf_path)
        results = {
            "doc_id": pdf_path.name,
            "full_path": str(pdf_path),
            "total_pages": len(doc),
            "pages": [],
            "processing_info": {
                "gpu_id": gpu_id,
                "device": device,
                "confidence_threshold": confidence_threshold,
                "group_headers": group_headers
            }
        }
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            headers = detector.detect_headers(
                page, 
                confidence_threshold=confidence_threshold, 
                group_headers=group_headers, 
                line_tolerance=line_tolerance, 
                word_gap_tolerance=word_gap_tolerance, 
                multiline=multiline
            )
            
            page_result = {
                "page_number": page_num + 1,
                "detected_headers": headers,
                "header_count": len(headers)
            }
            results["pages"].append(page_result)
        
        # Save results
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        doc.close()
        
        total_headers = sum(page["header_count"] for page in results["pages"])
        return f"Processed {pdf_path.name}: {total_headers} headers across {len(doc)} pages (GPU {gpu_id})"
        
    except Exception as e:
        error_msg = f"Error processing {pdf_path}: {e}"
        print(error_msg)
        return error_msg

def worker_init():
    """Initialize worker process"""
    # Set different random seeds for each worker to avoid identical behavior
    torch.manual_seed(os.getpid())
    np.random.seed(os.getpid())

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU header detection from final_groundtruth.csv")
    parser.add_argument("--csv_path", default="final_experiments/final_groundtruth.csv", 
                       help="Path to final_groundtruth.csv")
    parser.add_argument("--output_dir", default="header_detection_results", 
                       help="Directory to save results")
    parser.add_argument("--confidence", type=float, default=0.7, 
                       help="Confidence threshold (default: 0.5)")
    parser.add_argument("--num_gpus", type=int, default=6, 
                       help="Number of GPUs to use (default: 6)")
    parser.add_argument("--max_workers", type=int, 
                       help="Maximum number of worker processes (default: num_gpus)")
    parser.add_argument("--group_headers", action="store_true", default=True,
                       help="Group nearby header words into complete titles")
    parser.add_argument("--multiline", action="store_true", 
                       help="Allow grouping across multiple lines")
    parser.add_argument("--line_tolerance", type=int, default=100, 
                       help="Vertical distance to group words on same line")
    parser.add_argument("--word_gap", type=int, default=100, 
                       help="Max horizontal gap to group words")
    parser.add_argument("--limit", type=int, 
                       help="Limit number of PDFs to process (for testing)")
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU processing")
        args.num_gpus = 1
    else:
        actual_gpu_count = torch.cuda.device_count()
        if args.num_gpus > actual_gpu_count:
            print(f"Requested {args.num_gpus} GPUs, but only {actual_gpu_count} available")
            args.num_gpus = actual_gpu_count
    
    max_workers = args.max_workers or args.num_gpus
    
    # Load CSV file
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    
    # Get unique PDF paths
    unique_pdfs = df['full_path'].unique().tolist()
    if args.limit:
        unique_pdfs = unique_pdfs[:args.limit]
    
    print(f"Found {len(unique_pdfs)} unique PDFs to process")
    print(f"Using {args.num_gpus} GPUs with {max_workers} workers")
    print(f"Output directory: {args.output_dir}")
    
    # Prepare tasks with GPU assignments
    tasks = []
    for i, pdf_path in enumerate(unique_pdfs):
        gpu_id = i % args.num_gpus
        tasks.append((
            pdf_path, 
            args.output_dir, 
            args.confidence, 
            args.group_headers,
            args.line_tolerance, 
            args.word_gap, 
            args.multiline, 
            gpu_id
        ))
    
    # Process PDFs in parallel
    start_time = time.time()
    completed_count = 0
    error_count = 0
    
    with ProcessPoolExecutor(max_workers=max_workers, initializer=worker_init) as executor:
        # Submit all tasks
        future_to_pdf = {
            executor.submit(process_single_pdf, *task): task[0] 
            for task in tasks
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(tasks), desc="Processing PDFs") as pbar:
            for future in as_completed(future_to_pdf):
                pdf_path = future_to_pdf[future]
                try:
                    result = future.result()
                    if result.startswith("Error"):
                        error_count += 1
                        tqdm.write(result)
                    else:
                        completed_count += 1
                        tqdm.write(result)
                except Exception as exc:
                    error_count += 1
                    tqdm.write(f'PDF {pdf_path} generated an exception: {exc}')
                finally:
                    pbar.update(1)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\nProcessing completed!")
    print(f"Successfully processed: {completed_count}/{len(unique_pdfs)} PDFs")
    print(f"Errors: {error_count}")
    print(f"Total processing time: {processing_time:.2f} seconds")
    print(f"Average time per PDF: {processing_time/len(unique_pdfs):.2f} seconds")
    
    # Save summary
    summary = {
        "total_pdfs": len(unique_pdfs),
        "successful": completed_count,
        "errors": error_count,
        "processing_time_seconds": processing_time,
        "settings": {
            "confidence_threshold": args.confidence,
            "group_headers": args.group_headers,
            "num_gpus": args.num_gpus,
            "max_workers": max_workers
        }
    }
    
    summary_file = Path(args.output_dir) / "processing_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Start method may already be set
        pass
    
    main()