#!/usr/bin/env python3
"""
Create content files for missed conclusion pages using existing JSON data.
"""

import json
import PyPDF2
import fitz  # pymupdf - better for text extraction

def extract_pdf_text_advanced(pdf_path, page_num):
    """Extract text from a specific page using pymupdf (better quality)."""
    try:
        # Try pymupdf first (better text extraction)
        doc = fitz.open(pdf_path)
        if page_num <= len(doc):
            page = doc[page_num - 1]  # pymupdf uses 0-based indexing
            text = page.get_text()
            doc.close()
            return text.strip()
        doc.close()
    except Exception as e1:
        try:
            # Fallback to PyPDF2
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                if page_num <= len(pdf_reader.pages):
                    page = pdf_reader.pages[page_num - 1]  # PyPDF2 uses 0-based indexing
                    text = page.extract_text()
                    return text.strip()
        except Exception as e2:
            print(f"Error extracting text from {pdf_path} page {page_num}: {e1}, {e2}")
            return f"[ERROR: Could not extract text from page {page_num}]"
    
    return f"[ERROR: Page {page_num} not found in PDF]"

def create_conclusion_content_files():
    """Create content files using existing JSON data."""
    print("Creating conclusion content files from existing analysis...")
    
    # Load existing analysis
    with open('6_conclusion_pages_missed_detailed_analysis.json', 'r') as f:
        missed_pages = json.load(f)
    
    print(f"Found {len(missed_pages)} missed conclusion pages")
    
    # Filter for high miss rate pages (70%+)
    high_miss_pages = [page for page in missed_pages if page['miss_rate'] >= 0.7]
    
    # Create main content analysis file
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("MISSED CONCLUSION PAGES CONTENT ANALYSIS")
    output_lines.append("Pages correctly detected as chapter beginnings but missed as conclusions")
    output_lines.append("=" * 80)
    output_lines.append("")
    
    for i, page_info in enumerate(high_miss_pages[:20], 1):  # Top 20
        pdf_name = page_info['pdf_name']
        page_num = page_info['page_number']
        chapter_title = page_info.get('chapter_title', 'Unknown')
        label = page_info.get('label', 'Unknown')
        miss_rate = page_info['miss_rate']
        
        output_lines.append("=" * 80)
        output_lines.append(f"MISSED CONCLUSION PAGE #{i}")
        output_lines.append(f"PDF: {page_info['full_path']} - Page: {page_num}")
        output_lines.append("-" * 80)
        output_lines.append(f"Chapter Title: {chapter_title}")
        output_lines.append(f"Ground Truth Label: {label}")
        output_lines.append(f"Conclusion Miss Rate: {miss_rate:.2%} ({page_info['miss_count']}/{page_info['total_models']} models)")
        output_lines.append("")
        
        # Show model classifications from all_classifications if available
        if 'all_classifications' in page_info:
            output_lines.append("MODEL CLASSIFICATIONS:")
            for model_stage, classifications in page_info['all_classifications'].items():
                output_lines.append(f"  {model_stage}:")
                if isinstance(classifications, dict):
                    output_lines.append(f"    Chapter Detection: {classifications.get('chapter_classification', 'N/A')}")
                    output_lines.append(f"    Conclusion Detected: {classifications.get('conclusion_detected', 'N/A')}")
                else:
                    output_lines.append(f"    Status: {classifications}")
        output_lines.append("")
        
        # Extract the actual text content (first 500 characters)
        pdf_path = page_info['full_path']
        text_content = extract_pdf_text_advanced(pdf_path, page_num)
        
        output_lines.append("PAGE CONTENT:")
        output_lines.append("-" * 40)
        output_lines.append(text_content[:500] + ("..." if len(text_content) > 500 else ""))
        output_lines.append("")
        output_lines.append("")
    
    # Write main content file
    with open('14_missed_conclusion_pages_content_analysis.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    # Create all-missed file (100% miss rate)
    pages_missed_by_all = [page for page in missed_pages if page['miss_rate'] == 1.0]
    
    output_lines_all = []
    output_lines_all.append("=" * 80)
    output_lines_all.append("CONCLUSION PAGES MISSED BY ALL MODELS")
    output_lines_all.append("Pages correctly detected as chapter beginnings but ALL models missed as conclusions")
    output_lines_all.append("=" * 80)
    output_lines_all.append("")
    
    for i, page_info in enumerate(pages_missed_by_all, 1):
        pdf_name = page_info['pdf_name']
        page_num = page_info['page_number']
        chapter_title = page_info.get('chapter_title', 'Unknown')
        label = page_info.get('label', 'Unknown')
        
        output_lines_all.append(f"CONCLUSION PAGE #{i} - MISSED BY ALL")
        output_lines_all.append(f"PDF: {page_info['full_path']} - Page: {page_num}")
        output_lines_all.append(f"Chapter Title: {chapter_title}")
        output_lines_all.append(f"Ground Truth Label: {label}")
        output_lines_all.append("")
        
        # Show model classifications if available
        if 'all_classifications' in page_info:
            output_lines_all.append("MODEL CLASSIFICATIONS:")
            for model_stage, classifications in page_info['all_classifications'].items():
                output_lines_all.append(f"  {model_stage}:")
                if isinstance(classifications, dict):
                    output_lines_all.append(f"    Chapter Detection: {classifications.get('chapter_classification', 'N/A')}")
                    output_lines_all.append(f"    Conclusion Detected: {classifications.get('conclusion_detected', 'N/A')}")
                else:
                    output_lines_all.append(f"    Status: {classifications}")
        output_lines_all.append("")
        
        # Extract the actual text content (first 500 characters)
        pdf_path = page_info['full_path']
        text_content = extract_pdf_text_advanced(pdf_path, page_num)
        
        output_lines_all.append("PAGE CONTENT:")
        output_lines_all.append("-" * 40)
        output_lines_all.append(text_content[:500] + ("..." if len(text_content) > 500 else ""))
        output_lines_all.append("")
        output_lines_all.append("=" * 60)
        output_lines_all.append("")
    
    # Write all-missed content file
    with open('15_conclusion_pages_missed_by_all_content.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines_all))
    
    print(f"Created content files:")
    print(f"  - 14_missed_conclusion_pages_content_analysis.txt ({len(high_miss_pages[:20])} pages)")
    print(f"  - 15_conclusion_pages_missed_by_all_content.txt ({len(pages_missed_by_all)} pages)")

if __name__ == "__main__":
    create_conclusion_content_files()