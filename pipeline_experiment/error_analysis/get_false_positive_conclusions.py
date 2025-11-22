#!/usr/bin/env python3
"""
Get FALSE POSITIVE conclusion pages for each model individually using the CORRECT logic.

FALSE POSITIVES = Pages that are:
1. Correctly detected as chapter beginnings (intersection with ground truth)
2. NOT conclusions in ground truth 
3. BUT were detected as conclusions by the model

This identifies what non-conclusion pages are being misclassified as conclusions.
"""

import os
import re
import pandas as pd
from collections import defaultdict, Counter
import json

def extract_pages_from_file(file_path):
    """Extract page numbers from chapter_beginnings.txt or conclusion_beginnings.txt file."""
    if not os.path.exists(file_path):
        return {}
    
    pdf_pages = defaultdict(set)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
        # Find all PDF entries with page numbers
        pattern = r'PDF: ([^-]+\.pdf) - Page: (\d+)'
        matches = re.findall(pattern, content)
        
        for pdf_path, page_num in matches:
            # Extract just the filename from the full path
            pdf_filename = os.path.basename(pdf_path)
            pdf_pages[pdf_filename].add(int(page_num))
    
    return dict(pdf_pages)


def load_ground_truth(csv_path):
    """Load ground truth data from CSV file."""
    df = pd.read_csv(csv_path)
    
    all_ground_truth = defaultdict(set)
    conclusion_ground_truth = defaultdict(set)
    page_details = {}
    
    for _, row in df.iterrows():
        pdf_path = row['full_path']
        pdf_filename = os.path.basename(pdf_path)
        page_num = int(row['page_number'])
        label = row.get('label', 'Unknown')
        chapter_title = row.get('chapter_title', 'Unknown')
        
        # Add to all ground truth
        all_ground_truth[pdf_filename].add(page_num)
        
        # Add to conclusion ground truth if it contains CONCLUSION
        if pd.notna(label) and 'CONCLUSION' in str(label).upper():
            conclusion_ground_truth[pdf_filename].add(page_num)
        
        page_details[(pdf_filename, page_num)] = {
            'label': label,
            'chapter_title': chapter_title,
            'full_path': pdf_path
        }
    
    return dict(all_ground_truth), dict(conclusion_ground_truth), page_details


def get_false_positive_conclusions_for_model(results_dir, model_name, experiment_dir, stage_name, 
                                           all_ground_truth, conclusion_ground_truth, page_details):
    """Get false positive conclusions (FP) for a specific model/stage using correct logic."""
    
    # Build paths to detection files
    if model_name == experiment_dir:  # Direct experiment folder
        chapter_file = os.path.join(results_dir, model_name, stage_name, 'chapter_beginnings.txt')
        conclusion_file = os.path.join(results_dir, model_name, stage_name, 'conclusion_beginnings.txt')
    else:  # Model folder contains experiment folder
        chapter_file = os.path.join(results_dir, model_name, experiment_dir, stage_name, 'chapter_beginnings.txt')
        conclusion_file = os.path.join(results_dir, model_name, experiment_dir, stage_name, 'conclusion_beginnings.txt')
    
    if not os.path.exists(chapter_file) or not os.path.exists(conclusion_file):
        return []
    
    # Load detection results
    chapter_detections = extract_pages_from_file(chapter_file)
    conclusion_detections = extract_pages_from_file(conclusion_file)
    
    false_positive_conclusions = []
    
    # For each PDF in ground truth
    for pdf_name in all_ground_truth.keys():
        detected_chapters = chapter_detections.get(pdf_name, set())
        gt_chapters = all_ground_truth[pdf_name]
        
        # Find correctly detected chapter pages (intersection)
        correct_chapter_pages = detected_chapters.intersection(gt_chapters)
        
        # Among correctly detected chapter pages, find false positive conclusions
        for page_num in correct_chapter_pages:
            # Is this page actually a conclusion in ground truth?
            is_conclusion_gt = page_num in conclusion_ground_truth.get(pdf_name, set())
            
            # Was this page detected as a conclusion?
            is_conclusion_detected = page_num in conclusion_detections.get(pdf_name, set())
            
            # FALSE POSITIVE: Is NOT conclusion in GT but WAS detected as conclusion
            if not is_conclusion_gt and is_conclusion_detected:
                page_info = page_details.get((pdf_name, page_num), {})
                false_positive_conclusions.append({
                    'pdf_name': pdf_name,
                    'page_number': page_num,
                    'chapter_title': page_info.get('chapter_title', 'Unknown'),
                    'label': page_info.get('label', 'Unknown'),
                    'full_path': page_info.get('full_path', 'Unknown')
                })
    
    return false_positive_conclusions


def analyze_all_models_fp(results_dir, ground_truth_csv):
    """Analyze all models and get false positive conclusions for each."""
    
    print("Loading ground truth data...")
    all_ground_truth, conclusion_ground_truth, page_details = load_ground_truth(ground_truth_csv)
    
    # Find all model experiments
    model_experiments = []
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path):
            # Skip llama32Results
            if 'llama32' in item.lower():
                continue
            
            if item.startswith('pipeline_experiments_'):
                model_experiments.append((item, item, ''))
            else:
                for exp_item in os.listdir(item_path):
                    exp_path = os.path.join(item_path, exp_item)
                    if os.path.isdir(exp_path) and exp_item.startswith('pipeline_experiments_'):
                        model_experiments.append((item, exp_item, item))
    
    target_stages = ['stage2_3', 'full_pipeline']
    
    # Get false positive conclusions for each model
    all_fp_by_model = {}
    
    print("\\nAnalyzing false positive conclusions for each model...")
    
    for model_name, experiment_dir, model_folder in model_experiments:
        for stage in target_stages:
            model_stage_key = f"{model_folder if model_folder else model_name}_{stage}"
            print(f"Processing {model_stage_key}...")
            
            fp_conclusions = get_false_positive_conclusions_for_model(
                results_dir, model_name, experiment_dir, stage, 
                all_ground_truth, conclusion_ground_truth, page_details
            )
            
            all_fp_by_model[model_stage_key] = fp_conclusions
            print(f"  Found {len(fp_conclusions)} false positive conclusion pages (FP)")
    
    return all_fp_by_model, page_details


def compile_fp_commonality_analysis(all_fp_by_model):
    """Compile commonality analysis for false positives from individual model results."""
    print("\\nCompiling false positive commonality analysis...")
    
    # Count how many models have FP for each page
    page_fp_counts = Counter()
    total_models = len(all_fp_by_model)
    
    for model_stage_key, fp_pages in all_fp_by_model.items():
        for page in fp_pages:
            page_key = (page['pdf_name'], page['page_number'])
            page_fp_counts[page_key] += 1
    
    # Create commonality analysis
    fp_commonality_analysis = []
    
    for (pdf_name, page_num), fp_count in page_fp_counts.most_common():
        fp_rate = fp_count / total_models
        
        # Get page details from any model that had FP for it
        page_info = None
        for fp_pages in all_fp_by_model.values():
            for page in fp_pages:
                if page['pdf_name'] == pdf_name and page['page_number'] == page_num:
                    page_info = page
                    break
            if page_info:
                break
        
        if not page_info:
            continue
        
        # Determine which models had FP vs correct classification
        fp_by = []
        correct_by = []
        
        for model_stage_key, fp_pages in all_fp_by_model.items():
            found_in_fp = any(p['pdf_name'] == pdf_name and p['page_number'] == page_num 
                             for p in fp_pages)
            
            if found_in_fp:
                fp_by.append(model_stage_key)
            else:
                correct_by.append(model_stage_key)
        
        fp_commonality_analysis.append({
            'pdf_name': pdf_name,
            'page_number': page_num,
            'chapter_title': page_info['chapter_title'],
            'label': page_info['label'], 
            'full_path': page_info['full_path'],
            'fp_count': fp_count,
            'total_models': total_models,
            'fp_rate': fp_rate,
            'fp_by': fp_by,
            'correct_by': correct_by
        })
    
    return fp_commonality_analysis


def main():
    # Configuration - use same paths as existing scripts
    results_dir = '../results'
    ground_truth_csv = '../../final_groundtruth_filtered.csv'
    
    print("Starting analysis of FALSE POSITIVE conclusion pages by model...")
    print("Using the same logic as calculate_conclusion_metrics_on_correct_chapters.py")
    
    # Get all false positive conclusions for each model individually
    all_fp_by_model, page_details = analyze_all_models_fp(results_dir, ground_truth_csv)
    
    # Save individual model results
    print("\\nSaving individual model FP results...")
    with open('false_positive_conclusions_by_model.json', 'w') as f:
        json.dump(all_fp_by_model, f, indent=2)
    
    # Create summary table by model
    model_summary = []
    for model_stage_key, fp_pages in all_fp_by_model.items():
        model_summary.append({
            'Model_Stage': model_stage_key,
            'Total_False_Positive_Conclusions_FP': len(fp_pages)
        })
    
    model_df = pd.DataFrame(model_summary)
    print("\\nFalse positive conclusions (FP) by model:")
    print(model_df.to_string(index=False))
    model_df.to_csv('false_positive_conclusions_summary.csv', index=False)
    
    # Compile commonality analysis
    fp_commonality_analysis = compile_fp_commonality_analysis(all_fp_by_model)
    
    # Save detailed commonality results
    with open('fp_conclusion_commonality_analysis.json', 'w') as f:
        json.dump(fp_commonality_analysis, f, indent=2)
    
    # Create commonality summary tables
    print("\\n" + "="*100)
    print("FALSE POSITIVE CONCLUSION PAGES COMMONALITY ANALYSIS")
    print("="*100)
    
    # Pages with FP by all models
    fp_by_all = [p for p in fp_commonality_analysis if p['fp_rate'] == 1.0]
    print(f"\\nPages with FP by ALL {len(all_fp_by_model)} models: {len(fp_by_all)}")
    
    if fp_by_all:
        fp_all_df = pd.DataFrame([{
            'PDF': p['pdf_name'],
            'Page': p['page_number'], 
            'Chapter_Title': p['chapter_title'],
            'Label': p['label']
        } for p in fp_by_all])
        print(fp_all_df.to_string(index=False))
        fp_all_df.to_csv('fp_conclusions_by_all_models.csv', index=False)
    
    # Pages with FP by most models (70%+)
    fp_by_most = [p for p in fp_commonality_analysis if p['fp_rate'] >= 0.7]
    print(f"\\nPages with FP by 70%+ of models: {len(fp_by_most)}")
    
    if fp_by_most:
        fp_most_df = pd.DataFrame([{
            'PDF': p['pdf_name'],
            'Page': p['page_number'],
            'Chapter_Title': p['chapter_title'], 
            'Label': p['label'],
            'FP_Rate': f"{p['fp_rate']:.2%}",
            'FP_Count': f"{p['fp_count']}/{p['total_models']}"
        } for p in fp_by_most])
        print(fp_most_df.to_string(index=False))
        fp_most_df.to_csv('fp_conclusions_by_most_models.csv', index=False)
    
    print(f"\\nFalse Positive analysis complete! Numbers should match FP counts in metrics CSV.")
    print(f"Generated files:")
    print(f"  - false_positive_conclusions_by_model.json")
    print(f"  - false_positive_conclusions_summary.csv")
    print(f"  - fp_conclusion_commonality_analysis.json") 
    print(f"  - fp_conclusions_by_all_models.csv")
    print(f"  - fp_conclusions_by_most_models.csv")


if __name__ == "__main__":
    main()