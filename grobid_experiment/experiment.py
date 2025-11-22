#!/usr/bin/env python3

"""
Simple section header detection using LayoutLMv3 DocLayNet model.
Just extracts section headers from PDFs - no NLI, no classification, no preprocessing.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

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
        
        print(f"Available labels: {list(self.id2label.values())}")
        
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
                      group_headers=False, line_tolerance=20, word_gap_tolerance=50, multiline=False):
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
            
            # Always return both original and grouped for comparison
            if group_headers:
                grouped = group_nearby_headers(detected_headers, line_tolerance, word_gap_tolerance, multiline)
                return {
                    "original": detected_headers,
                    "grouped": grouped
                }
            else:
                return detected_headers
            
        except Exception as e:
            print(f"Error processing page: {e}")
            return []

def process_pdf(pdf_path: Path, detector: SectionHeaderDetector, confidence_threshold=0.5, 
                output_dir: Optional[Path] = None, group_headers=False, line_tolerance=20, word_gap_tolerance=50, multiline=False):
    """Process a single PDF and extract section headers"""
    
    try:
        doc = fitz.open(pdf_path)
        results = {
            "doc_id": pdf_path.name,
            "total_pages": len(doc),
            "pages": []
        }
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            headers_result = detector.detect_headers(page, confidence_threshold=confidence_threshold, 
                                                   group_headers=group_headers, line_tolerance=line_tolerance, 
                                                   word_gap_tolerance=word_gap_tolerance, multiline=multiline)
            
            if group_headers and isinstance(headers_result, dict):
                # We have both original and grouped
                original_headers = headers_result["original"]
                grouped_headers = headers_result["grouped"]
                
                # Analyze grouping effectiveness
                print(f"Page {page_num + 1}:")
                print(f"  Original: {len(original_headers)} headers")
                for i, h in enumerate(original_headers):
                    y_center = (h['bbox'][1] + h['bbox'][3]) / 2
                    print(f"    {i}: '{h['text']}' y_center={y_center:.1f} bbox={h['bbox']}")
                
                print(f"  Grouped: {len(grouped_headers)} headers")
                for i, h in enumerate(grouped_headers):
                    y_center = (h['bbox'][1] + h['bbox'][3]) / 2
                    word_count = h.get('word_count', 1)
                    print(f"    {i}: '{h['text']}' y_center={y_center:.1f} words={word_count}")
                
                # Check specific case: CHAPTER + number + title
                if len(original_headers) >= 3:
                    h1, h2, h3 = original_headers[0], original_headers[1], original_headers[2]
                    y1 = (h1['bbox'][1] + h1['bbox'][3]) / 2
                    y2 = (h2['bbox'][1] + h2['bbox'][3]) / 2  
                    y3 = (h3['bbox'][1] + h3['bbox'][3]) / 2
                    
                    v_dist_12 = abs(y2 - y1)
                    v_dist_23 = abs(y3 - y2)
                    v_dist_13 = abs(y3 - y1)
                    
                    print(f"  Analysis:")
                    print(f"    '{h1['text']}' to '{h2['text']}': v_dist={v_dist_12:.1f}")
                    print(f"    '{h2['text']}' to '{h3['text']}': v_dist={v_dist_23:.1f}")
                    print(f"    '{h1['text']}' to '{h3['text']}': v_dist={v_dist_13:.1f}")
                    print(f"    Tolerance: {line_tolerance}")
                
                page_result = {
                    "page_number": page_num + 1,
                    "original_headers": original_headers,
                    "detected_headers": grouped_headers,
                    "header_count": len(grouped_headers),
                    "original_count": len(original_headers)
                }
            else:
                # Regular format
                page_result = {
                    "page_number": page_num + 1,
                    "detected_headers": headers_result,
                    "header_count": len(headers_result)
                }
            results["pages"].append(page_result)
            
            # # Print progress
            # if headers:
            #     print(f"Page {page_num + 1}: Found {len(headers)} headers")
            #     for header in headers:
            #         print(f"  - '{header['text']}' (confidence: {header['confidence']:.3f})")
        
        # Save detailed results if output directory specified
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{pdf_path.stem}_headers.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Saved detailed results to {output_file}")
        
        doc.close()
        return results
        
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return {
            "doc_id": pdf_path.name,
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="Simple section header detection with LayoutLMv3")
    parser.add_argument("--pdf_dir", required=True, help="Directory containing PDFs")
    parser.add_argument("--pdf_file", help="Single PDF file to process")
    parser.add_argument("--output_dir", help="Directory to save detailed results")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    parser.add_argument("--limit", type=int, help="Limit number of PDFs to process")
    parser.add_argument("--device", help="Device to use (cuda/cpu)")
    parser.add_argument("--group_headers", action="store_true", help="Group nearby header words into complete titles")
    parser.add_argument("--multiline", action="store_true", help="Allow grouping across multiple lines (ignores horizontal gaps)")
    parser.add_argument("--line_tolerance", type=int, default=20, help="Vertical distance to group words on same line (pixels)")
    parser.add_argument("--word_gap", type=int, default=50, help="Max horizontal gap to group words (pixels)")
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = SectionHeaderDetector(device=args.device)
    
    # Get PDF files to process
    if args.pdf_file:
        pdf_files = [Path(args.pdf_file)]
    else:
        pdf_dir = Path(args.pdf_dir)
        pdf_files = list(pdf_dir.rglob("*.pdf"))
        pdf_files.sort()
        
        if args.limit:
            pdf_files = pdf_files[:args.limit]
    
    if not pdf_files:
        print("No PDF files found!")
        sys.exit(1)
    
    print(f"Processing {len(pdf_files)} PDF files...")
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    all_results = []
    
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        result = process_pdf(pdf_path, detector, args.confidence, output_dir, 
                           args.group_headers, args.line_tolerance, args.word_gap, args.multiline)
        all_results.append(result)
        print("-" * 50)
    
    # Summary
    total_headers = 0
    successful_pdfs = 0
    
    for result in all_results:
        if "error" not in result:
            successful_pdfs += 1
            for page in result["pages"]:
                total_headers += page["header_count"]
    
    print(f"\nSUMMARY:")
    print(f"Successfully processed: {successful_pdfs}/{len(pdf_files)} PDFs")
    print(f"Total section headers detected: {total_headers}")
    print(f"Average headers per PDF: {total_headers/max(1, successful_pdfs):.1f}")

if __name__ == "__main__":
    main()