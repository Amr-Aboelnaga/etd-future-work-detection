#!/usr/bin/env python3
"""
Extract the actual 500-character content that models see for both:
1. False Negatives (missed conclusions)
2. False Positives (incorrectly identified as conclusions)

This will help analyze what content patterns lead to errors.
"""

import os
import json
import pandas as pd
import fitz  # pymupdf for PDF text extraction

def extract_pdf_text_advanced(pdf_path, page_num, char_limit=500):
    """Extract text from a specific page using pymupdf."""
    try:
        doc = fitz.open(pdf_path)
        if page_num <= len(doc):
            page = doc[page_num - 1]  # pymupdf uses 0-based indexing
            text = page.get_text()
            doc.close()
            # Limit to first 500 characters (what models actually see)
            return text.strip()[:char_limit] if text.strip() else "[EMPTY PAGE]"
        doc.close()
    except Exception as e:
        return f"[ERROR: Could not extract text from page {page_num}: {e}]"
    
    return f"[ERROR: Page {page_num} not found in PDF]"


def load_error_pages(json_file):
    """Load error pages from JSON file."""
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        return {}


def extract_content_for_errors(error_data, error_type, output_file):
    """Extract content for all error pages across all models."""
    print(f"\\nExtracting content for {error_type} pages...")
    
    # Collect all unique pages across all models
    all_error_pages = {}
    
    for model_stage, pages in error_data.items():
        for page in pages:
            page_key = (page['pdf_name'], page['page_number'])
            if page_key not in all_error_pages:
                all_error_pages[page_key] = {
                    'pdf_name': page['pdf_name'],
                    'page_number': page['page_number'],
                    'chapter_title': page['chapter_title'],
                    'label': page['label'],
                    'full_path': page['full_path'],
                    'models_with_error': []
                }
            all_error_pages[page_key]['models_with_error'].append(model_stage)
    
    # Extract content for each page
    content_analysis = []
    
    for i, ((pdf_name, page_num), page_info) in enumerate(all_error_pages.items(), 1):
        print(f"  Processing page {i}/{len(all_error_pages)}: {pdf_name}:{page_num}")
        
        # Extract the actual 500-character content
        pdf_path = page_info['full_path']
        content_500 = extract_pdf_text_advanced(pdf_path, page_num, 500)
        
        # Count how many models have this error
        error_count = len(page_info['models_with_error'])
        total_models = 8  # Total number of model configurations
        error_rate = error_count / total_models
        
        content_analysis.append({
            'pdf_name': pdf_name,
            'page_number': page_num,
            'chapter_title': page_info['chapter_title'],
            'label': page_info['label'],
            'full_path': page_info['full_path'],
            'error_count': error_count,
            'total_models': total_models,
            'error_rate': error_rate,
            'models_with_error': page_info['models_with_error'],
            'content_500_chars': content_500
        })
    
    # Sort by error rate (most common errors first)
    content_analysis.sort(key=lambda x: x['error_rate'], reverse=True)
    
    # Save detailed analysis
    with open(f'{error_type}_content_analysis.json', 'w') as f:
        json.dump(content_analysis, f, indent=2)
    
    # Create text file with readable content analysis
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"{'='*100}\\n")
        f.write(f"{error_type.upper()} PAGES - CONTENT ANALYSIS\\n")
        f.write(f"Pages with {error_type.lower()} sorted by frequency across models\\n")
        f.write(f"{'='*100}\\n\\n")
        
        for i, page in enumerate(content_analysis, 1):
            f.write(f"{'='*80}\\n")
            f.write(f"{error_type.upper()} PAGE #{i}\\n")
            f.write(f"PDF: {page['pdf_name']} - Page: {page['page_number']}\\n")
            f.write(f"{'='*80}\\n")
            f.write(f"Chapter Title: {page['chapter_title']}\\n")
            f.write(f"Ground Truth Label: {page['label']}\\n")
            f.write(f"Error Rate: {page['error_rate']:.1%} ({page['error_count']}/{page['total_models']} models)\\n")
            f.write(f"\\nModels with {error_type.lower()}:\\n")
            for model in page['models_with_error']:
                f.write(f"  - {model}\\n")
            f.write(f"\\nACTUAL 500-CHARACTER CONTENT THAT MODELS SEE:\\n")
            f.write(f"{'-'*50}\\n")
            f.write(f"{page['content_500_chars']}\\n")
            f.write(f"{'-'*50}\\n\\n")
            
            # Add separator for very long files
            if i % 10 == 0:
                f.write(f"\\n{'#'*100}\\n")
                f.write(f"# PAGES {i-9}-{i} COMPLETE - CONTINUING...\\n") 
                f.write(f"{'#'*100}\\n\\n")
    
    return content_analysis


def create_summary_tables(fn_analysis, fp_analysis):
    """Create summary tables for patterns."""
    print("\\nCreating summary tables...")
    
    # Pages missed by ALL models (100% error rate)
    fn_by_all = [p for p in fn_analysis if p['error_rate'] == 1.0]
    fp_by_all = [p for p in fp_analysis if p['error_rate'] == 1.0] 
    
    print(f"\\nFalse Negatives by ALL models: {len(fn_by_all)}")
    if fn_by_all:
        fn_all_df = pd.DataFrame([{
            'PDF': p['pdf_name'],
            'Page': p['page_number'],
            'Chapter_Title': p['chapter_title'],
            'Label': p['label'],
            'Content_Preview': p['content_500_chars'][:100] + "..." if len(p['content_500_chars']) > 100 else p['content_500_chars']
        } for p in fn_by_all])
        print(fn_all_df.to_string(index=False))
        fn_all_df.to_csv('fn_by_all_models_with_content.csv', index=False)
    
    print(f"\\nFalse Positives by ALL models: {len(fp_by_all)}")
    if fp_by_all:
        fp_all_df = pd.DataFrame([{
            'PDF': p['pdf_name'],
            'Page': p['page_number'],
            'Chapter_Title': p['chapter_title'], 
            'Label': p['label'],
            'Content_Preview': p['content_500_chars'][:100] + "..." if len(p['content_500_chars']) > 100 else p['content_500_chars']
        } for p in fp_by_all])
        print(fp_all_df.to_string(index=False))
        fp_all_df.to_csv('fp_by_all_models_with_content.csv', index=False)
    
    # Pages with high error rates (70%+)
    fn_high = [p for p in fn_analysis if p['error_rate'] >= 0.7]
    fp_high = [p for p in fp_analysis if p['error_rate'] >= 0.7]
    
    print(f"\\nHigh-frequency False Negatives (70%+): {len(fn_high)}")
    print(f"High-frequency False Positives (70%+): {len(fp_high)}")


def main():
    print("Starting content extraction for error analysis...")
    
    # Load error data from the correct analysis files
    fn_data = load_error_pages('correct_missed_conclusions_by_model.json')
    fp_data = load_error_pages('false_positive_conclusions_by_model.json')
    
    if not fn_data:
        print("ERROR: Could not load False Negative data")
        return
    
    if not fp_data:
        print("ERROR: Could not load False Positive data") 
        return
    
    print(f"Loaded FN data for {len(fn_data)} model configurations")
    print(f"Loaded FP data for {len(fp_data)} model configurations")
    
    # Extract content for False Negatives (missed conclusions)
    fn_analysis = extract_content_for_errors(
        fn_data, 
        "FALSE_NEGATIVE",
        "missed_conclusions_content_analysis.txt"
    )
    
    # Extract content for False Positives (wrong conclusions)  
    fp_analysis = extract_content_for_errors(
        fp_data,
        "FALSE_POSITIVE", 
        "false_positive_conclusions_content_analysis.txt"
    )
    
    # Create summary tables
    create_summary_tables(fn_analysis, fp_analysis)
    
    print(f"\\nContent extraction complete!")
    print(f"Generated files:")
    print(f"  - missed_conclusions_content_analysis.txt")
    print(f"  - FALSE_NEGATIVE_content_analysis.json") 
    print(f"  - false_positive_conclusions_content_analysis.txt")
    print(f"  - FALSE_POSITIVE_content_analysis.json")
    print(f"  - fn_by_all_models_with_content.csv")
    print(f"  - fp_by_all_models_with_content.csv")
    
    print(f"\\nNow you can analyze:")
    print(f"  - What content patterns cause ALL models to miss conclusions")
    print(f"  - What content patterns cause ALL models to incorrectly identify conclusions")
    print(f"  - Model-specific vs universal failure patterns")


if __name__ == "__main__":
    main()