#!/usr/bin/env python3
"""
Get ALL missed conclusion pages (FN) for each model individually using the CORRECT logic
from calculate_conclusion_metrics_on_correct_chapters.py, then compile commonality analysis.

The correct logic:
1. Find pages that were correctly detected as chapter beginnings (intersection with ground truth)
2. Among those pages, find which are conclusions in ground truth but NOT detected as conclusions
3. Those are the FALSE NEGATIVES = missed conclusions
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


def get_missed_conclusions_for_model(results_dir, model_name, experiment_dir, stage_name, 
                                   all_ground_truth, conclusion_ground_truth, page_details):
    """Get missed conclusions (FN) for a specific model/stage using correct logic."""
    
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
    
    missed_conclusions = []
    
    # For each PDF in ground truth
    for pdf_name in all_ground_truth.keys():
        detected_chapters = chapter_detections.get(pdf_name, set())
        gt_chapters = all_ground_truth[pdf_name]
        
        # Find correctly detected chapter pages (intersection)
        correct_chapter_pages = detected_chapters.intersection(gt_chapters)
        
        # Among correctly detected chapter pages, find missed conclusions
        for page_num in correct_chapter_pages:
            # Is this page actually a conclusion in ground truth?
            is_conclusion_gt = page_num in conclusion_ground_truth.get(pdf_name, set())
            
            # Was this page detected as a conclusion?
            is_conclusion_detected = page_num in conclusion_detections.get(pdf_name, set())
            
            # FALSE NEGATIVE: Is conclusion in GT but NOT detected as conclusion
            if is_conclusion_gt and not is_conclusion_detected:
                page_info = page_details.get((pdf_name, page_num), {})
                missed_conclusions.append({
                    'pdf_name': pdf_name,
                    'page_number': page_num,
                    'chapter_title': page_info.get('chapter_title', 'Unknown'),
                    'label': page_info.get('label', 'Unknown'),
                    'full_path': page_info.get('full_path', 'Unknown')
                })
    
    return missed_conclusions


def analyze_all_models(results_dir, ground_truth_csv):
    """Analyze all models and get missed conclusions for each."""
    
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
    
    # Get missed conclusions for each model
    all_missed_by_model = {}
    
    print("\\nAnalyzing missed conclusions for each model...")
    
    for model_name, experiment_dir, model_folder in model_experiments:
        for stage in target_stages:
            model_stage_key = f"{model_folder if model_folder else model_name}_{stage}"
            print(f"Processing {model_stage_key}...")
            
            missed_conclusions = get_missed_conclusions_for_model(
                results_dir, model_name, experiment_dir, stage, 
                all_ground_truth, conclusion_ground_truth, page_details
            )
            
            all_missed_by_model[model_stage_key] = missed_conclusions
            print(f"  Found {len(missed_conclusions)} missed conclusion pages (FN)")
    
    return all_missed_by_model, page_details


def compile_commonality_analysis(all_missed_by_model):
    """Compile commonality analysis from individual model results."""
    print("\\nCompiling commonality analysis...")
    
    # Count how many models miss each page
    page_miss_counts = Counter()
    total_models = len(all_missed_by_model)
    
    for model_stage_key, missed_pages in all_missed_by_model.items():
        for page in missed_pages:
            page_key = (page['pdf_name'], page['page_number'])
            page_miss_counts[page_key] += 1
    
    # Create commonality analysis
    commonality_analysis = []
    
    for (pdf_name, page_num), miss_count in page_miss_counts.most_common():
        miss_rate = miss_count / total_models
        
        # Get page details from any model that missed it
        page_info = None
        for missed_pages in all_missed_by_model.values():
            for page in missed_pages:
                if page['pdf_name'] == pdf_name and page['page_number'] == page_num:
                    page_info = page
                    break
            if page_info:
                break
        
        if not page_info:
            continue
        
        # Determine which models missed vs detected
        missed_by = []
        detected_by = []
        
        for model_stage_key, missed_pages in all_missed_by_model.items():
            found_in_missed = any(p['pdf_name'] == pdf_name and p['page_number'] == page_num 
                                 for p in missed_pages)
            
            if found_in_missed:
                missed_by.append(model_stage_key)
            else:
                detected_by.append(model_stage_key)
        
        commonality_analysis.append({
            'pdf_name': pdf_name,
            'page_number': page_num,
            'chapter_title': page_info['chapter_title'],
            'label': page_info['label'], 
            'full_path': page_info['full_path'],
            'miss_count': miss_count,
            'total_models': total_models,
            'miss_rate': miss_rate,
            'missed_by': missed_by,
            'detected_by': detected_by
        })
    
    return commonality_analysis


def main():
    # Configuration - use same paths as existing scripts
    results_dir = '../results'
    ground_truth_csv = '../../final_groundtruth_filtered.csv'
    
    print("Starting CORRECT analysis of missed conclusion pages by model...")
    print("Using the same logic as calculate_conclusion_metrics_on_correct_chapters.py")
    
    # Get all missed conclusions for each model individually
    all_missed_by_model, page_details = analyze_all_models(results_dir, ground_truth_csv)
    
    # Save individual model results
    print("\\nSaving individual model results...")
    with open('correct_missed_conclusions_by_model.json', 'w') as f:
        json.dump(all_missed_by_model, f, indent=2)
    
    # Create summary table by model
    model_summary = []
    for model_stage_key, missed_pages in all_missed_by_model.items():
        model_summary.append({
            'Model_Stage': model_stage_key,
            'Total_Missed_Conclusions_FN': len(missed_pages)
        })
    
    model_df = pd.DataFrame(model_summary)
    print("\\nMissed conclusions (FN) by model:")
    print(model_df.to_string(index=False))
    model_df.to_csv('correct_missed_conclusions_summary.csv', index=False)
    
    # Compile commonality analysis
    commonality_analysis = compile_commonality_analysis(all_missed_by_model)
    
    # Save detailed commonality results
    with open('correct_conclusion_commonality_analysis.json', 'w') as f:
        json.dump(commonality_analysis, f, indent=2)
    
    # Create commonality summary tables
    print("\\n" + "="*100)
    print("CORRECT CONCLUSION PAGES COMMONALITY ANALYSIS")
    print("="*100)
    
    # Pages missed by all models
    missed_by_all = [p for p in commonality_analysis if p['miss_rate'] == 1.0]
    print(f"\\nPages missed by ALL {len(all_missed_by_model)} models: {len(missed_by_all)}")
    
    if missed_by_all:
        missed_all_df = pd.DataFrame([{
            'PDF': p['pdf_name'],
            'Page': p['page_number'], 
            'Chapter_Title': p['chapter_title'],
            'Label': p['label']
        } for p in missed_by_all])
        print(missed_all_df.to_string(index=False))
        missed_all_df.to_csv('correct_conclusions_missed_by_all_models.csv', index=False)
    
    # Pages missed by most models (70%+)
    missed_by_most = [p for p in commonality_analysis if p['miss_rate'] >= 0.7]
    print(f"\\nPages missed by 70%+ of models: {len(missed_by_most)}")
    
    if missed_by_most:
        missed_most_df = pd.DataFrame([{
            'PDF': p['pdf_name'],
            'Page': p['page_number'],
            'Chapter_Title': p['chapter_title'], 
            'Label': p['label'],
            'Miss_Rate': f"{p['miss_rate']:.2%}",
            'Missed_Count': f"{p['miss_count']}/{p['total_models']}"
        } for p in missed_by_most])
        print(missed_most_df.to_string(index=False))
        missed_most_df.to_csv('correct_conclusions_missed_by_most_models.csv', index=False)
    
    print(f"\\nCorrect analysis complete! Numbers should match FN counts in metrics CSV.")
    print(f"Generated files:")
    print(f"  - correct_missed_conclusions_by_model.json")
    print(f"  - correct_missed_conclusions_summary.csv")
    print(f"  - correct_conclusion_commonality_analysis.json") 
    print(f"  - correct_conclusions_missed_by_all_models.csv")
    print(f"  - correct_conclusions_missed_by_most_models.csv")


if __name__ == "__main__":
    main()