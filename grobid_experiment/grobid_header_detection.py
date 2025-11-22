#!/usr/bin/env python3

"""
GROBID-based header detection script that processes PDFs from final_groundtruth.csv
using GROBID for structure parsing and header extraction.
Outputs the same JSON format as multi_gpu_header_detection.py for compatibility
with existing classification scripts.
"""

import argparse
import json
import pandas as pd
import requests
import re
from lxml import etree
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import sys
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

import fitz  # PyMuPDF for page mapping
from tqdm import tqdm


class GROBIDHeaderDetector:
    def __init__(self, grobid_url: str = "http://localhost:8070/api"):
        """Initialize GROBID client"""
        self.grobid_url = grobid_url
        self.session = requests.Session()
        
        # Test GROBID connection
        try:
            response = self.session.get(f"{self.grobid_url}/isalive", timeout=10)
            if response.status_code != 200:
                raise Exception(f"GROBID not responding: {response.status_code}")
            print(f"Connected to GROBID at {self.grobid_url}")
        except Exception as e:
            raise Exception(f"Cannot connect to GROBID at {grobid_url}: {e}")
    
    def process_pdf(self, pdf_path: Path) -> str:
        """Process PDF with GROBID and return TEI XML"""
        try:
            with open(pdf_path, 'rb') as pdf_file:
                files = {'input': pdf_file}
                data = {
                    'consolidateHeader': '1',  # Enable header consolidation
                    'includeRawCitations': '1',  # Include raw citations
                    'teiCoordinates': ['head']  # Add coordinates for headers
                }
                response = self.session.post(
                    f"{self.grobid_url}/processFulltextDocument",
                    files=files,
                    data=data,
                    timeout=300  # 5 minutes timeout
                )
                response.raise_for_status()
                return response.text
        except Exception as e:
            raise Exception(f"GROBID processing failed: {e}")
    
    def extract_headers_from_tei(self, tei_xml: str, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract headers from GROBID TEI XML and map to PDF pages"""
        try:
            # Parse TEI XML with lxml
            root = etree.fromstring(tei_xml.encode('utf-8'))
            
            # Define namespaces
            ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
            
            # Extract headers from the structure
            headers = []
            
            # Find all div elements (sections) in the body
            body = root.find('.//tei:body', ns)
            if body is not None:
                # Find all div elements with head children (section headers)
                for div in body.findall('.//tei:div', ns):
                    head = div.find('tei:head', ns)
                    if head is not None:
                        header_text = self._extract_text_content(head)
                        if header_text.strip():
                            # Get coordinates if available
                            coords = self._extract_coordinates(head)
                            
                            
                            header_info = {
                                'text': header_text.strip(),
                                'coords': coords,
                                'level': self._get_header_level(div),
                                'element': 'head'
                            }
                            headers.append(header_info)
            
            # Also check for headers in front matter (abstract, keywords, etc.)
            front = root.find('.//tei:front', ns)
            if front is not None:
                for div in front.findall('.//tei:div', ns):
                    head = div.find('tei:head', ns)
                    if head is not None:
                        header_text = self._extract_text_content(head)
                        if header_text.strip():
                            coords = self._extract_coordinates(head)
                            
                            
                            header_info = {
                                'text': header_text.strip(),
                                'coords': coords,
                                'level': 0,  # Front matter headers
                                'element': 'head'
                            }
                            headers.append(header_info)
            
            # Map headers to PDF pages using coordinates
            page_headers = self._map_headers_to_pages(headers, pdf_path)
            
            return page_headers
            
        except etree.XMLSyntaxError as e:
            print(f"XML parsing error: {e}")
            return []
        except Exception as e:
            print(f"Header extraction error: {e}")
            return []
    
    def _extract_text_content(self, element) -> str:
        """Recursively extract text content from XML element"""
        text = element.text or ""
        for child in element:
            text += self._extract_text_content(child)
            if child.tail:
                text += child.tail
        return text
    
    def _extract_coordinates(self, element) -> Optional[List[Dict[str, float]]]:
        """Extract bounding box coordinates from GROBID element
        
        GROBID format: "page,x,y,width,height[;page,x,y,width,height]"
        Returns list of bounding boxes (for multi-line headers)
        """
        coords_attr = element.get('coords')
        if not coords_attr:
            return None
            
        try:
            # Split multiple coordinate boxes (separated by ;)
            coord_boxes = []
            for box_coords in coords_attr.split(';'):
                coords = [float(x) for x in box_coords.split(',')]
                if len(coords) == 5:  # page,x,y,width,height format
                    page, x, y, width, height = coords
                    coord_boxes.append({
                        'page': int(page),
                        'x': x, 'y': y, 'width': width, 'height': height,
                        # Convert to standard bbox format (x0,y0,x1,y1)
                        'x0': x, 'y0': y, 'x1': x + width, 'y1': y + height
                    })
            return coord_boxes if coord_boxes else None
        except (ValueError, IndexError):
            return None
    
    def _get_header_level(self, div_element) -> int:
        """Determine header level from div nesting"""
        level = 1
        parent = div_element.getparent()
        while parent is not None and parent.tag.endswith('}div'):
            level += 1
            parent = parent.getparent()
        return level
    
    def _map_headers_to_pages(self, headers: List[Dict], pdf_path: Path) -> List[Dict[str, Any]]:
        """Map headers to PDF pages using GROBID coordinates"""
        try:
            # Open PDF to get total pages
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            doc.close()
            
            # Initialize page headers list
            page_headers = [[] for _ in range(total_pages)]
            
            for header_info in headers:
                header_text = header_info['text']
                coords = header_info.get('coords')
                
                # Skip headers without coordinates
                if not coords:
                    print(f"Warning: No coordinates for header '{header_text[:50]}...'")
                    continue
                
                # Process each coordinate box (for multi-line headers)
                for coord_box in coords:
                    page_num = coord_box['page'] - 1  # Convert to 0-based indexing
                    
                    # Validate page number
                    if page_num < 0 or page_num >= total_pages:
                        print(f"Warning: Invalid page {page_num + 1} for header '{header_text[:50]}...' (total pages: {total_pages})")
                        continue
                    
                    # Calculate confidence based on text length and coordinates presence
                    confidence = min(0.95, 0.7 + len(header_text.split()) * 0.05)
                    
                    # Filter headers (keep substantial headers or section markers)
                    if len(header_text.split()) >= 2 or self._is_section_marker(header_text):
                        detected_header = {
                            'text': header_text,
                            'bbox': [coord_box['x0'], coord_box['y0'], coord_box['x1'], coord_box['y1']],
                            'confidence': confidence,
                            'word_count': len(header_text.split()),
                            'level': header_info.get('level', 1),
                            'source': 'grobid_coords',
                            'grobid_page': coord_box['page']  # Original page number from GROBID
                        }
                        page_headers[page_num].append(detected_header)
                        print(f"Mapped header '{header_text}' to page {page_num + 1}")
            
            # Sort headers on each page by vertical position (y-coordinate)
            for page_idx in range(total_pages):
                page_headers[page_idx].sort(key=lambda h: h['bbox'][1])
            
            return page_headers
            
        except Exception as e:
            print(f"Error mapping headers to pages: {e}")
            return []
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', text.lower())).strip()
    
    def _is_section_marker(self, text: str) -> bool:
        """Check if text looks like a section marker (e.g., 'Chapter 1', 'Section 2.1')"""
        patterns = [
            r'^\d+\.?\d*\.?\s*$',  # Just numbers
            r'^(chapter|section|part)\s+\d+',  # Chapter/Section + number
            r'^\d+\.\d+',  # Numbered sections like 2.1
            r'^[ivxlc]+\.?\s*$',  # Roman numerals
        ]
        
        norm_text = text.lower().strip()
        return any(re.match(pattern, norm_text) for pattern in patterns)


def process_single_pdf_grobid(pdf_path: str, output_dir: str, grobid_url: str = "http://localhost:8070/api", 
                              worker_id: int = 0) -> str:
    """Process a single PDF using GROBID and extract headers"""
    
    try:
        detector = GROBIDHeaderDetector(grobid_url)
        
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if output already exists
        output_file = output_dir / f"{pdf_path.stem}.json"
        if output_file.exists():
            return f"Skipped {pdf_path.name} - output already exists"
        
        # Process PDF with GROBID
        tei_xml = detector.process_pdf(pdf_path)
        page_headers = detector.extract_headers_from_tei(tei_xml, pdf_path)
        
        # Open PDF to get total pages
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        
        # Format results to match expected structure
        results = {
            "doc_id": pdf_path.name,
            "full_path": str(pdf_path),
            "total_pages": total_pages,
            "pages": [],
            "processing_info": {
                "worker_id": worker_id,
                "grobid_url": grobid_url,
                "source": "grobid"
            }
        }
        
        # Ensure we have headers for each page
        for page_num in range(total_pages):
            if page_num < len(page_headers):
                detected_headers = page_headers[page_num]
            else:
                detected_headers = []
            
            page_result = {
                "page_number": page_num + 1,
                "detected_headers": detected_headers,
                "header_count": len(detected_headers)
            }
            results["pages"].append(page_result)
        
        # Save results
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        total_headers = sum(page["header_count"] for page in results["pages"])
        return f"Processed {pdf_path.name}: {total_headers} headers across {total_pages} pages (Worker {worker_id})"
        
    except Exception as e:
        error_msg = f"Error processing {pdf_path}: {e}"
        print(error_msg)
        return error_msg


def main():
    parser = argparse.ArgumentParser(description="GROBID-based header detection from final_groundtruth.csv")
    parser.add_argument("--csv_path", default="../final_groundtruth_filtered.csv", 
                       help="Path to final_groundtruth.csv")
    parser.add_argument("--output_dir", default="header_results", 
                       help="Directory to save results")
    parser.add_argument("--grobid_url", default="http://localhost:8070/api",
                       help="GROBID server URL")
    parser.add_argument("--max_workers", type=int, default=4,
                       help="Maximum number of worker processes")
    parser.add_argument("--limit", type=int, 
                       help="Limit number of PDFs to process (for testing)")
    
    args = parser.parse_args()
    
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
    print(f"Using {args.max_workers} workers")
    print(f"Output directory: {args.output_dir}")
    print(f"GROBID URL: {args.grobid_url}")
    
    # Test GROBID connection before starting
    try:
        test_detector = GROBIDHeaderDetector(args.grobid_url)
        print("✓ GROBID connection successful")
    except Exception as e:
        print(f"✗ GROBID connection failed: {e}")
        print("Please ensure GROBID server is running at the specified URL")
        sys.exit(1)
    
    # Prepare tasks
    tasks = []
    for i, pdf_path in enumerate(unique_pdfs):
        tasks.append((pdf_path, args.output_dir, args.grobid_url, i % args.max_workers))
    
    # Process PDFs in parallel
    start_time = time.time()
    completed_count = 0
    error_count = 0
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        future_to_pdf = {
            executor.submit(process_single_pdf_grobid, *task): task[0] 
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
            "grobid_url": args.grobid_url,
            "max_workers": args.max_workers,
            "source": "grobid"
        }
    }
    
    summary_file = Path(args.output_dir) / "processing_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    # Set multiprocessing start method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Start method may already be set
        pass
    
    main()