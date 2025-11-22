#!/usr/bin/env python3
"""
CORRECTED VERSION: Extract actual text content of truly missed pages.

CORRECT LOGIC:
1. STAGE 1 - Chapter Beginning Detection:
   - TRUE MISS: Ground truth page NOT classified as "Chapter beginning page"
   - TRUE HIT: Ground truth page classified as "Chapter beginning page"

2. STAGE 2 - Conclusion Classification (only on HITS from Stage 1):
   - Among pages correctly detected as chapter beginnings, check conclusion detection
   - This is handled separately in the conclusion detection analysis

This script focuses ONLY on Stage 1 failures - pages that should have been detected 
as chapter beginnings but were classified as something else.
"""

import os
import re
import pandas as pd
import json
from collections import defaultdict
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


def load_ground_truth():
    """Load ground truth data from CSV file."""
    df = pd.read_csv('../../final_groundtruth_filtered.csv')
    
    ground_truth = defaultdict(set)
    page_details = {}
    
    for _, row in df.iterrows():
        pdf_path = row['full_path']
        pdf_filename = os.path.basename(pdf_path)
        page_num = int(row['page_number'])
        label = row.get('label', 'Unknown')
        chapter_title = row.get('chapter_title', 'Unknown')
        
        ground_truth[pdf_filename].add(page_num)
        page_details[(pdf_filename, page_num)] = {
            'label': label,
            'chapter_title': chapter_title,
            'full_path': pdf_path
        }
    
    return dict(ground_truth), page_details


def get_model_classifications(pdf_name, page_num, results_dir):
    """
    Get what each model actually classified a ground truth page as.
    CORRECT LOGIC: Look at JSON analysis files to see actual classification.
    """
    classifications = {}
    
    # Option to skip llama32Results
    skip_llama32 = True
    
    # Find all model experiments
    model_experiments = []
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path):
            # Skip llama32Results if requested
            if skip_llama32 and 'llama32' in item.lower():
                continue
            
            if item.startswith('pipeline_experiments_'):
                model_experiments.append((item, item, ''))
            else:
                for exp_item in os.listdir(item_path):
                    exp_path = os.path.join(item_path, exp_item)
                    if os.path.isdir(exp_path) and exp_item.startswith('pipeline_experiments_'):
                        model_experiments.append((item, exp_item, item))
    
    target_stages = ['stage2_3', 'full_pipeline']
    
    for model_name, experiment_dir, model_folder in model_experiments:
        for stage in target_stages:
            model_stage_key = f"{model_folder if model_folder else model_name}_{stage}"
            classifications[model_stage_key] = "NOT_FOUND"
            
            # Build path to the stage directory
            if model_folder:
                stage_dir = os.path.join(results_dir, model_folder, experiment_dir, stage)
            else:
                stage_dir = os.path.join(results_dir, model_name, stage)
            
            if not os.path.exists(stage_dir):
                continue
            
            # Look for JSON analysis files that match this PDF
            pdf_base_name = pdf_name.replace('.pdf', '')
            
            # Try different possible JSON file naming patterns
            possible_json_files = [
                f"{pdf_base_name}_analysis.json",
                f"{pdf_name}_analysis.json",
                # Some files might have different naming patterns
                f"{pdf_base_name.replace('_', '-')}_analysis.json",
                f"{pdf_base_name.replace('-', '_')}_analysis.json"
            ]
            
            classification_found = False
            for json_filename in possible_json_files:
                json_path = os.path.join(stage_dir, json_filename)
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            analysis_data = json.load(f)
                        
                        # Look through the pages to find this specific page number
                        if 'pages' in analysis_data:
                            for page_data in analysis_data['pages']:
                                if page_data.get('page_number') == page_num:
                                    category = page_data.get('category', 'UNKNOWN_CATEGORY')
                                    classifications[model_stage_key] = category
                                    classification_found = True
                                    break
                        
                        if classification_found:
                            break
                            
                    except (json.JSONDecodeError, KeyError, Exception) as e:
                        print(f"Error reading {json_path}: {e}")
                        continue
    
    return classifications


def find_truly_missed_chapter_pages():
    """
    CORRECTED LOGIC: Find pages that are in ground truth but were NOT classified as "Chapter beginning page"
    These are the ACTUAL misses for Stage 1 (Chapter Beginning Detection)
    """
    print("Finding truly missed chapter beginning pages...")
    
    # Load ground truth
    ground_truth, page_details = load_ground_truth()
    results_dir = '../results'
    
    # Option to skip llama32Results
    skip_llama32 = True
    
    # Find all model experiments
    model_experiments = []
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path):
            # Skip llama32Results if requested
            if skip_llama32 and 'llama32' in item.lower():
                continue
            
            if item.startswith('pipeline_experiments_'):
                model_experiments.append((item, item, ''))
            else:
                for exp_item in os.listdir(item_path):
                    exp_path = os.path.join(item_path, exp_item)
                    if os.path.isdir(exp_path) and exp_item.startswith('pipeline_experiments_'):
                        model_experiments.append((item, exp_item, item))
    
    target_stages = ['stage2_3', 'full_pipeline']
    total_model_stages = len(model_experiments) * len(target_stages)
    
    # Track missed pages for each model/stage combination
    all_missed_pages = []
    
    print(f"Analyzing {len(model_experiments)} model experiments...")
    
    for model_name, experiment_dir, model_folder in model_experiments:
        print(f"Processing {model_name}")
        
        for stage in target_stages:
            model_stage_key = f"{model_folder if model_folder else model_name}_{stage}"
            
            # For each ground truth page, check what this model classified it as
            for pdf_name, gt_pages in ground_truth.items():
                for page_num in gt_pages:
                    # Get what this model classified this page as
                    classifications = get_model_classifications(pdf_name, page_num, results_dir)
                    actual_classification = classifications.get(model_stage_key, "NOT_FOUND")
                    
                    # CORRECT LOGIC: It's a miss if it's NOT classified as "Chapter beginning page"
                    if actual_classification != "Chapter beginning page":
                        all_missed_pages.append({
                            'pdf_name': pdf_name,
                            'page_number': page_num,
                            'model': model_folder if model_folder else model_name,
                            'stage': stage,
                            'model_stage': model_stage_key,
                            'actual_classification': actual_classification
                        })
    
    # Count how many times each page was truly missed
    from collections import Counter
    page_miss_counts = Counter()
    for missed in all_missed_pages:
        page_key = (missed['pdf_name'], missed['page_number'])
        page_miss_counts[page_key] += 1
    
    # Create detailed analysis of truly missed pages
    truly_missed_pages = []
    
    for (pdf_name, page_num), miss_count in page_miss_counts.most_common():
        miss_rate = miss_count / total_model_stages
        page_info = page_details.get((pdf_name, page_num), {})
        
        # Get all the actual classifications for this page
        all_classifications = {}
        for missed in all_missed_pages:
            if missed['pdf_name'] == pdf_name and missed['page_number'] == page_num:
                all_classifications[missed['model_stage']] = missed['actual_classification']
        
        truly_missed_pages.append({
            'pdf_name': pdf_name,
            'page_number': page_num,
            'chapter_title': page_info.get('chapter_title', 'Unknown'),
            'label': page_info.get('label', 'Unknown'),
            'full_path': page_info.get('full_path', 'Unknown'),
            'miss_count': miss_count,
            'total_models': total_model_stages,
            'miss_rate': miss_rate,
            'all_classifications': all_classifications
        })
    
    return truly_missed_pages


def extract_truly_missed_pages_content():
    """Extract content for pages that were truly missed in chapter beginning detection."""
    print("Extracting content for truly missed chapter beginning pages...")
    
    truly_missed_pages = find_truly_missed_chapter_pages()
    
    # Focus on pages missed by most models (70%+ miss rate)
    high_miss_pages = [page for page in truly_missed_pages if page['miss_rate'] >= 0.7]
    
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("TRULY MISSED CHAPTER BEGINNING PAGES - CORRECTED ANALYSIS")
    output_lines.append("Pages that should be 'Chapter beginning page' but were classified as something else")
    output_lines.append("=" * 80)
    output_lines.append("")
    
    for i, page_info in enumerate(high_miss_pages[:20], 1):  # Top 20 most missed
        pdf_name = page_info['pdf_name']
        page_num = page_info['page_number']
        chapter_title = page_info.get('chapter_title', 'Unknown')
        label = page_info.get('label', 'Unknown')
        miss_rate = page_info['miss_rate']
        
        output_lines.append("=" * 80)
        output_lines.append(f"TRULY MISSED PAGE #{i}")
        output_lines.append(f"PDF: {page_info['full_path']} - Page: {page_num}")
        output_lines.append("-" * 80)
        output_lines.append(f"Chapter Title: {chapter_title}")
        output_lines.append(f"Ground Truth Label: {label}")
        output_lines.append(f"Miss Rate: {miss_rate:.2%} ({page_info['miss_count']}/{page_info['total_models']} models)")
        output_lines.append("")
        
        # Show what each model actually classified this page as
        output_lines.append("ACTUAL MODEL CLASSIFICATIONS (should be 'Chapter beginning page'):")
        for model_stage, classification in page_info['all_classifications'].items():
            output_lines.append(f"  {model_stage}: {classification}")
        output_lines.append("")
        
        # Extract the actual text content
        pdf_path = page_info['full_path']
        text_content = extract_pdf_text_advanced(pdf_path, page_num)
        
        output_lines.append("PAGE CONTENT:")
        output_lines.append("-" * 40)
        output_lines.append(text_content[:500] + ("..." if len(text_content) > 500 else ""))
        output_lines.append("")
        output_lines.append("")
    
    # Write to file
    with open('12_missed_chapter_pages_content_analysis.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    print(f"Extracted content for {len(high_miss_pages[:20])} truly missed chapter pages")
    
    return truly_missed_pages


def extract_pages_missed_by_all_corrected():
    """Extract pages that were missed by ALL models in chapter beginning detection."""
    print("Extracting pages truly missed by ALL models...")
    
    truly_missed_pages = find_truly_missed_chapter_pages()
    
    # Pages missed by all models (100% miss rate) 
    pages_missed_by_all = [page for page in truly_missed_pages if page['miss_rate'] == 1.0]
    
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("CHAPTER PAGES TRULY MISSED BY ALL MODELS - CORRECTED ANALYSIS")
    output_lines.append("Pages that should be 'Chapter beginning page' but ALL models classified as something else")
    output_lines.append("=" * 80)
    output_lines.append("")
    
    for i, page_info in enumerate(pages_missed_by_all, 1):
        pdf_name = page_info['pdf_name']
        page_num = page_info['page_number']
        chapter_title = page_info.get('chapter_title', 'Unknown')
        label = page_info.get('label', 'Unknown')
        
        output_lines.append(f"CHAPTER PAGE #{i} - TRULY MISSED BY ALL")
        output_lines.append(f"PDF: {page_info['full_path']} - Page: {page_num}")
        output_lines.append(f"Chapter Title: {chapter_title}")
        output_lines.append(f"Ground Truth Label: {label}")
        output_lines.append("")
        
        # Show what each model actually classified this page as
        output_lines.append("ACTUAL MODEL CLASSIFICATIONS (should be 'Chapter beginning page'):")
        for model_stage, classification in page_info['all_classifications'].items():
            output_lines.append(f"  {model_stage}: {classification}")
        output_lines.append("")
        
        # Extract the actual text content
        pdf_path = page_info['full_path']
        text_content = extract_pdf_text_advanced(pdf_path, page_num)
        
        output_lines.append("PAGE CONTENT:")
        output_lines.append("-" * 40)
        output_lines.append(text_content[:500] + ("..." if len(text_content) > 500 else ""))
        output_lines.append("")
        output_lines.append("=" * 60)
        output_lines.append("")
    
    # Write to file
    with open('13_chapter_pages_missed_by_all_content.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    print(f"Extracted content for {len(pages_missed_by_all)} chapter pages truly missed by ALL models")
    
    return pages_missed_by_all


def main():
    """Main function with corrected logic."""
    print("Starting CORRECTED extraction of missed pages content...")
    print("CORRECT LOGIC: Only pages NOT classified as 'Chapter beginning page' are considered missed")
    print("")
    
    # Extract truly missed pages
    extract_truly_missed_pages_content()
    extract_pages_missed_by_all_corrected()
    
    print("\nContent extraction completed!")
    print("\nGenerated files:")
    print("  - 12_missed_chapter_pages_content_analysis.txt")
    print("  - 13_chapter_pages_missed_by_all_content.txt")
    print("\nThese files contain pages that were ACTUALLY missed in chapter beginning detection.")


if __name__ == "__main__":
    main()