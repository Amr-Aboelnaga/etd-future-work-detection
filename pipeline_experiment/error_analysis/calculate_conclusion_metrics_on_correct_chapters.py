#!/usr/bin/env python3
"""
Calculate conclusion detection metrics ONLY on the pages that were correctly identified as chapter beginnings.
This measures the second-stage classification performance: given that we correctly found a chapter beginning,
how well do we classify whether it's a conclusion chapter or not?
"""

import os
import re
import pandas as pd
import json
from collections import defaultdict


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


def calculate_conclusion_metrics_on_correct_chapters(
    chapter_detections, conclusion_detections, all_ground_truth, conclusion_ground_truth
):
    """
    Calculate conclusion detection metrics only on correctly detected chapter pages.
    
    Logic:
    1. Find pages that were correctly detected as chapter beginnings (TP chapter pages)
    2. For those pages, check if they should be conclusions (ground truth)
    3. For those pages, check if they were detected as conclusions
    4. Calculate TP/FP/FN for conclusion classification on this subset
    """
    
    # Find all correctly detected chapter beginning pages
    correctly_detected_chapter_pages = []
    
    for pdf_name in all_ground_truth.keys():
        detected_chapters = chapter_detections.get(pdf_name, set())
        gt_chapters = all_ground_truth[pdf_name]
        
        # Pages that were correctly detected as chapter beginnings
        correct_chapter_pages = detected_chapters.intersection(gt_chapters)
        
        for page_num in correct_chapter_pages:
            correctly_detected_chapter_pages.append((pdf_name, page_num))
    
    # Now classify these correctly detected chapter pages for conclusions
    total_tp = 0  # Correctly detected chapters that are conclusions AND detected as conclusions
    total_fp = 0  # Correctly detected chapters that are NOT conclusions but detected as conclusions  
    total_fn = 0  # Correctly detected chapters that are conclusions but NOT detected as conclusions
    
    page_results = []
    
    for pdf_name, page_num in correctly_detected_chapter_pages:
        # Is this page actually a conclusion in ground truth?
        is_conclusion_gt = page_num in conclusion_ground_truth.get(pdf_name, set())
        
        # Was this page detected as a conclusion?
        is_conclusion_detected = page_num in conclusion_detections.get(pdf_name, set())
        
        if is_conclusion_gt and is_conclusion_detected:
            # True Positive: Correctly detected chapter that is a conclusion and was detected as conclusion
            total_tp += 1
            result_type = "TP"
        elif not is_conclusion_gt and is_conclusion_detected:
            # False Positive: Correctly detected chapter that is NOT a conclusion but was detected as conclusion
            total_fp += 1
            result_type = "FP"
        elif is_conclusion_gt and not is_conclusion_detected:
            # False Negative: Correctly detected chapter that is a conclusion but was NOT detected as conclusion
            total_fn += 1
            result_type = "FN"
        else:
            # True Negative: Correctly detected chapter that is NOT a conclusion and was NOT detected as conclusion
            result_type = "TN"
        
        page_results.append({
            'pdf_name': pdf_name,
            'page_num': page_num,
            'is_conclusion_gt': is_conclusion_gt,
            'is_conclusion_detected': is_conclusion_detected,
            'result_type': result_type
        })
    
    # Calculate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'total_correctly_detected_chapters': len(correctly_detected_chapter_pages),
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'total_tn': len(correctly_detected_chapter_pages) - total_tp - total_fp - total_fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'page_results': page_results
    }


def analyze_model_stage(results_dir, model_name, experiment_dir, stage_name, all_ground_truth, conclusion_ground_truth):
    """Analyze conclusion detection on correctly detected chapter pages for a specific model and stage."""
    
    # Build paths
    if experiment_dir:
        chapter_file = os.path.join(results_dir, model_name, experiment_dir, stage_name, 'chapter_beginnings.txt')
        conclusion_file = os.path.join(results_dir, model_name, experiment_dir, stage_name, 'conclusion_beginnings.txt')
    else:
        chapter_file = os.path.join(results_dir, model_name, stage_name, 'chapter_beginnings.txt')
        conclusion_file = os.path.join(results_dir, model_name, stage_name, 'conclusion_beginnings.txt')
    
    if not os.path.exists(chapter_file) or not os.path.exists(conclusion_file):
        return None
    
    # Extract detections
    chapter_detections = extract_pages_from_file(chapter_file)
    conclusion_detections = extract_pages_from_file(conclusion_file)
    
    # Calculate metrics only for correctly detected chapter pages
    metrics = calculate_conclusion_metrics_on_correct_chapters(
        chapter_detections, conclusion_detections, all_ground_truth, conclusion_ground_truth
    )
    
    # Add model/stage info
    metrics['model'] = model_name
    metrics['stage'] = stage_name
    metrics['experiment'] = experiment_dir if experiment_dir else model_name
    
    return metrics


def main():
    """Main function to calculate conclusion detection metrics on correctly detected chapter pages."""
    results_dir = '../results'
    ground_truth_file = '../../final_groundtruth_filtered.csv'
    
    # Option to skip llama32Results (set to False to include it)
    skip_llama32 = True
    
    # Load ground truth data
    print("Loading ground truth data...")
    all_ground_truth, conclusion_ground_truth, page_details = load_ground_truth(ground_truth_file)
    
    print(f"Loaded {len(all_ground_truth)} PDFs with {sum(len(pages) for pages in all_ground_truth.values())} total chapter pages")
    print(f"Of these, {len(conclusion_ground_truth)} PDFs have {sum(len(pages) for pages in conclusion_ground_truth.values())} conclusion pages")
    
    # Find all model experiments (excluding llama32Results)
    model_experiments = []
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path):
            # Skip llama32Results if requested
            if skip_llama32 and 'llama32' in item.lower():
                continue
            
            if item.startswith('pipeline_experiments_'):
                # Direct experiment folder
                model_experiments.append((item, item, ''))
            else:
                # Model-specific folder
                for exp_item in os.listdir(item_path):
                    exp_path = os.path.join(item_path, exp_item)
                    if os.path.isdir(exp_path) and exp_item.startswith('pipeline_experiments_'):
                        model_experiments.append((item, exp_item, item))
    
    all_results = []
    target_stages = ['stage2_3', 'full_pipeline']
    
    print(f"\nAnalyzing conclusion detection on correctly detected chapter pages...")
    
    for model_name, experiment_dir, model_folder in model_experiments:
        print(f"\nProcessing {model_name} - {experiment_dir}")
        
        for stage in target_stages:
            if model_folder:
                # Model-specific folder structure
                result = analyze_model_stage(results_dir, model_folder, experiment_dir, stage, all_ground_truth, conclusion_ground_truth)
            else:
                # Direct experiment folder  
                result = analyze_model_stage(results_dir, model_name, '', stage, all_ground_truth, conclusion_ground_truth)
                if result:
                    result['model'] = model_name
                    result['experiment'] = model_name
            
            if result:
                all_results.append(result)
                print(f"  {stage}: {result['total_correctly_detected_chapters']} correct chapters, "
                      f"TP={result['total_tp']}, FP={result['total_fp']}, FN={result['total_fn']}, "
                      f"P={result['precision']:.3f}, R={result['recall']:.3f}, F1={result['f1_score']:.3f}")
    
    # Output files will be saved in current directory (error_analysis)
    
    # Save detailed results
    with open('5_conclusion_detection_metrics_detailed.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Create summary report
    print("\n" + "="*120)
    print("CONCLUSION DETECTION METRICS ON CORRECTLY DETECTED CHAPTER PAGES")
    print("="*120)
    
    summary_data = []
    for result in all_results:
        summary_data.append({
            'Model': result['model'],
            'Stage': result['stage'],
            'Correct_Chapters': result['total_correctly_detected_chapters'],
            'TP': result['total_tp'],
            'FP': result['total_fp'],
            'FN': result['total_fn'],
            'TN': result['total_tn'],
            'Precision': f"{result['precision']:.3f}",
            'Recall': f"{result['recall']:.3f}",
            'F1_Score': f"{result['f1_score']:.3f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('5_conclusion_detection_metrics_on_detected_chapters.csv', index=False)
    
    print(summary_df.to_string(index=False))
    
    # Verify alignment with coverage analysis
    print(f"\n{'='*80}")
    print("ALIGNMENT VERIFICATION WITH COVERAGE ANALYSIS")
    print("='*80")
    
    coverage_expected = {
        'llama33Results_stage2_3': 288, 'llama33Results_full_pipeline': 290,
        'llama4ScoutResults_stage2_3': 273, 'llama4ScoutResults_full_pipeline': 275,
        'llama3.18bResults_stage2_3': 272, 'llama3.18bResults_full_pipeline': 283,
        'mistralsmallResults_stage2_3': 279, 'mistralsmallResults_full_pipeline': 278
    }
    
    print("Correctly detected chapters vs Expected from coverage:")
    for result in all_results:
        key = f"{result['model']}_{result['stage']}"
        expected = coverage_expected.get(key, "Unknown")
        actual = result['total_correctly_detected_chapters']
        status = "✓" if actual == expected else "✗"
        print(f"  {key:35} {actual:3d} vs {expected:3} {status}")
    
    print(f"\nResults saved to current directory:")
    print(f"  - 5_conclusion_detection_metrics_detailed.json")
    print(f"  - 5_conclusion_detection_metrics_on_detected_chapters.csv")


if __name__ == "__main__":
    main()