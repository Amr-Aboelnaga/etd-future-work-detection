#!/usr/bin/env python3
"""
Calculate ground truth coverage for chapter beginning detection.
Compares detected chapter beginnings with ground truth data for stage2_3 and full_pipeline.
"""

import os
import re
import pandas as pd
from collections import defaultdict
import json


def extract_chapter_pages_from_file(file_path):
    """Extract chapter beginning page numbers from chapter_beginnings.txt file."""
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
    
    ground_truth = defaultdict(set)
    
    for _, row in df.iterrows():
        pdf_path = row['full_path']
        pdf_filename = os.path.basename(pdf_path)
        page_num = row['page_number']
        ground_truth[pdf_filename].add(int(page_num))
    
    return dict(ground_truth)


def calculate_coverage(detected_pages, ground_truth_pages):
    """Calculate coverage metrics for a single PDF."""
    if not ground_truth_pages:
        return {
            'total_ground_truth': 0,
            'detected_ground_truth': 0,
            'coverage_rate': 0.0,
            'total_detected': len(detected_pages),
            'false_positives': len(detected_pages)
        }
    
    ground_truth_set = set(ground_truth_pages)
    detected_set = set(detected_pages)
    
    detected_ground_truth = ground_truth_set.intersection(detected_set)
    false_positives = detected_set - ground_truth_set
    
    coverage_rate = len(detected_ground_truth) / len(ground_truth_set) if ground_truth_set else 0.0
    
    return {
        'total_ground_truth': len(ground_truth_set),
        'detected_ground_truth': len(detected_ground_truth),
        'coverage_rate': coverage_rate,
        'total_detected': len(detected_set),
        'false_positives': len(false_positives),
        'detected_pages': detected_ground_truth,
        'missed_pages': ground_truth_set - detected_set,
        'false_positive_pages': false_positives
    }


def analyze_model_stage(results_dir, model_name, experiment_dir, stage_name, ground_truth):
    """Analyze a specific model and stage combination."""
    chapter_file = os.path.join(results_dir, model_name, experiment_dir, stage_name, 'chapter_beginnings.txt')
    
    if not os.path.exists(chapter_file):
        return None
    
    detected_pages = extract_chapter_pages_from_file(chapter_file)
    
    results = {
        'model': model_name,
        'stage': stage_name,
        'experiment': experiment_dir,
        'total_pdfs_detected': len(detected_pages),
        'total_pdfs_ground_truth': len(ground_truth),
        'pdfs_with_matches': 0,
        'overall_coverage': 0.0,
        'total_ground_truth_pages': 0,
        'total_detected_ground_truth_pages': 0,
        'total_false_positives': 0,
        'pdf_results': {}
    }
    
    total_ground_truth = 0
    total_detected_ground_truth = 0
    total_false_positives = 0
    pdfs_with_matches = 0
    
    # Analyze each PDF that has ground truth data
    for pdf_name, gt_pages in ground_truth.items():
        detected = detected_pages.get(pdf_name, set())
        coverage = calculate_coverage(detected, gt_pages)
        
        results['pdf_results'][pdf_name] = coverage
        
        total_ground_truth += coverage['total_ground_truth']
        total_detected_ground_truth += coverage['detected_ground_truth']
        total_false_positives += coverage['false_positives']
        
        if coverage['detected_ground_truth'] > 0:
            pdfs_with_matches += 1
    
    # Calculate overall metrics
    results['total_ground_truth_pages'] = total_ground_truth
    results['total_detected_ground_truth_pages'] = total_detected_ground_truth
    results['total_false_positives'] = total_false_positives
    results['overall_coverage'] = total_detected_ground_truth / total_ground_truth if total_ground_truth > 0 else 0.0
    results['pdfs_with_matches'] = pdfs_with_matches
    
    return results


def main():
    """Main function to calculate ground truth coverage."""
    results_dir = '../results'
    ground_truth_file = '../../final_groundtruth_filtered.csv'
    
    # Option to skip llama32Results (set to False to include it)
    skip_llama32 = True
    
    # Load ground truth data
    print("Loading ground truth data...")
    ground_truth = load_ground_truth(ground_truth_file)
    print(f"Loaded ground truth for {len(ground_truth)} PDFs with {sum(len(pages) for pages in ground_truth.values())} total pages")
    
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
    
    print(f"\nAnalyzing {len(model_experiments)} model experiments...")
    
    for model_name, experiment_dir, model_folder in model_experiments:
        print(f"\nProcessing {model_name} - {experiment_dir}")
        
        for stage in target_stages:
            if model_folder:
                # Model-specific folder structure
                result = analyze_model_stage(results_dir, model_folder, experiment_dir, stage, ground_truth)
            else:
                # Direct experiment folder
                result = analyze_model_stage(results_dir, model_name, '', stage, ground_truth)
                if result:
                    result['model'] = model_name
                    result['experiment'] = model_name
            
            if result:
                all_results.append(result)
                print(f"  {stage}: {result['overall_coverage']:.3f} coverage "
                      f"({result['total_detected_ground_truth_pages']}/{result['total_ground_truth_pages']} pages)")
    
    # Output files will be saved in current directory (error_analysis)
    
    # Save detailed results
    with open('1_chapter_beginning_detection_coverage_detailed.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Create summary report
    print("\n" + "="*80)
    print("GROUND TRUTH COVERAGE SUMMARY")
    print("="*80)
    
    summary_data = []
    for result in all_results:
        summary_data.append({
            'Model': result['model'],
            'Stage': result['stage'],
            'Coverage_Rate': f"{result['overall_coverage']:.3f}",
            'Detected_GT_Pages': result['total_detected_ground_truth_pages'],
            'Total_GT_Pages': result['total_ground_truth_pages'],
            'False_Positives': result['total_false_positives'],
            'PDFs_with_Matches': result['pdfs_with_matches']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('1_chapter_beginning_detection_coverage_by_model.csv', index=False)
    
    print(summary_df.to_string(index=False))
    
    # Best and worst performers
    print(f"\n{'Best Performers:':<20}")
    best = sorted(all_results, key=lambda x: x['overall_coverage'], reverse=True)[:3]
    for result in best:
        print(f"  {result['model']:20} {result['stage']:15} {result['overall_coverage']:.3f}")
    
    print(f"\n{'Worst Performers:':<20}")
    worst = sorted(all_results, key=lambda x: x['overall_coverage'])[:3]
    for result in worst:
        print(f"  {result['model']:20} {result['stage']:15} {result['overall_coverage']:.3f}")
    
    print(f"\nDetailed results saved to: 1_chapter_beginning_detection_coverage_detailed.json")
    print(f"Summary saved to: 1_chapter_beginning_detection_coverage_by_model.csv")


if __name__ == "__main__":
    main()