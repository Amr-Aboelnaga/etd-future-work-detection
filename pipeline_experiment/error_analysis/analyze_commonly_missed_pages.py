#!/usr/bin/env python3
"""
Analyze commonly missed ground truth pages across all models and stages.
Identifies pages that are frequently missed by most or all models.
"""

import os
import re
import pandas as pd
from collections import defaultdict, Counter
import json


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


def load_ground_truth(csv_path):
    """Load ground truth data from CSV file."""
    df = pd.read_csv(csv_path)
    
    ground_truth = defaultdict(set)
    page_details = {}
    
    for _, row in df.iterrows():
        pdf_path = row['full_path']
        pdf_filename = os.path.basename(pdf_path)
        page_num = row['page_number']
        chapter_title = row.get('chapter_title', 'Unknown')
        label = row.get('label', 'Unknown')
        
        ground_truth[pdf_filename].add(int(page_num))
        page_details[(pdf_filename, int(page_num))] = {
            'chapter_title': chapter_title,
            'label': label,
            'full_path': pdf_path
        }
    
    return dict(ground_truth), page_details


def analyze_missed_pages():
    """Analyze commonly missed ground truth pages across all models and stages."""
    results_dir = '../results'
    ground_truth_file = '../../final_groundtruth_filtered.csv'
    
    # Option to skip llama32Results (set to False to include it)
    skip_llama32 = True
    
    # Load ground truth data
    print("Loading ground truth data...")
    ground_truth, page_details = load_ground_truth(ground_truth_file)
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
    
    target_stages = ['stage2_3', 'full_pipeline']
    
    # Track missed pages for each model/stage combination
    all_missed_pages = []
    total_model_stages = len(model_experiments) * len(target_stages)
    
    print(f"\nAnalyzing missed pages across {len(model_experiments)} model experiments...")
    
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
    
    # Count how many times each page was missed
    page_miss_counts = Counter()
    for missed in all_missed_pages:
        page_key = (missed['pdf_name'], missed['page_number'])
        page_miss_counts[page_key] += 1
    
    # Create detailed analysis
    commonly_missed = []
    
    for (pdf_name, page_num), miss_count in page_miss_counts.most_common():
        miss_rate = miss_count / total_model_stages
        page_info = page_details.get((pdf_name, page_num), {})
        
        # Get which model/stages detected vs missed this page
        detected_by = []
        missed_by = []
        
        # Check each model/stage to see if it detected this page correctly
        for model_name, experiment_dir, model_folder in model_experiments:
            for stage in target_stages:
                model_stage_key = f"{model_folder if model_folder else model_name}_{stage}"
                
                # Get what this model classified this page as
                classifications = get_model_classifications(pdf_name, page_num, results_dir)
                actual_classification = classifications.get(model_stage_key, "NOT_FOUND")
                
                if actual_classification == "Chapter beginning page":
                    detected_by.append(model_stage_key)
                else:
                    missed_by.append(model_stage_key)
        
        commonly_missed.append({
            'pdf_name': pdf_name,
            'page_number': page_num,
            'chapter_title': page_info.get('chapter_title', 'Unknown'),
            'label': page_info.get('label', 'Unknown'),
            'full_path': page_info.get('full_path', 'Unknown'),
            'miss_count': miss_count,
            'total_models': total_model_stages,
            'miss_rate': miss_rate,
            'detected_by': detected_by,
            'missed_by': missed_by
        })
    
    # Output files will be saved in current directory (error_analysis)
    
    # Save detailed results
    with open('2_chapter_pages_missed_detailed_analysis.json', 'w') as f:
        json.dump(commonly_missed, f, indent=2)
    
    # Create summary tables
    
    # 1. Most commonly missed pages (missed by all or most models)
    print("\n" + "="*100)
    print("MOST COMMONLY MISSED GROUND TRUTH PAGES")
    print("="*100)
    
    most_missed = [p for p in commonly_missed if p['miss_rate'] >= 0.7]  # Missed by 70%+ of models
    most_missed_df = pd.DataFrame([{
        'PDF': p['pdf_name'][:30] + '...' if len(p['pdf_name']) > 30 else p['pdf_name'],
        'Page': p['page_number'],
        'Chapter_Title': p['chapter_title'][:50] + '...' if len(str(p['chapter_title'])) > 50 else p['chapter_title'],
        'Label': p['label'],
        'Miss_Rate': f"{p['miss_rate']:.2f}",
        'Missed_Count': f"{p['miss_count']}/{p['total_models']}"
    } for p in most_missed[:20]])
    
    print(most_missed_df.to_string(index=False))
    most_missed_df.to_csv('2_chapter_pages_missed_by_most_models.csv', index=False)
    
    # 2. Pages missed by all models
    print(f"\n{'='*60}")
    print("PAGES MISSED BY ALL MODELS/STAGES")
    print("="*60)
    
    missed_by_all = [p for p in commonly_missed if p['miss_rate'] == 1.0]
    print(f"Total pages missed by all {total_model_stages} model/stage combinations: {len(missed_by_all)}")
    
    if missed_by_all:
        missed_all_df = pd.DataFrame([{
            'PDF': p['pdf_name'],
            'Page': p['page_number'],
            'Chapter_Title': p['chapter_title'],
            'Label': p['label']
        } for p in missed_by_all])
        
        print(missed_all_df.to_string(index=False))
        missed_all_df.to_csv('3_chapter_pages_missed_by_all_models.csv', index=False)
    
    # 3. Label-based analysis
    print(f"\n{'='*60}")
    print("MISSED PAGES BY LABEL TYPE")
    print("="*60)
    
    label_miss_stats = defaultdict(list)
    for page in commonly_missed:
        label_miss_stats[page['label']].append(page['miss_rate'])
    
    label_summary = []
    for label, miss_rates in label_miss_stats.items():
        label_summary.append({
            'Label': label,
            'Total_Pages': len(miss_rates),
            'Avg_Miss_Rate': f"{sum(miss_rates) / len(miss_rates):.3f}",
            'Pages_Missed_By_All': len([r for r in miss_rates if r == 1.0]),
            'Pages_Missed_By_Most': len([r for r in miss_rates if r >= 0.7])
        })
    
    label_df = pd.DataFrame(label_summary)
    label_df = label_df.sort_values('Avg_Miss_Rate', ascending=False)
    print(label_df.to_string(index=False))
    label_df.to_csv('4_chapter_pages_miss_rates_by_label_type.csv', index=False)
    
    # 4. Model performance comparison
    print(f"\n{'='*60}")
    print("MODEL/STAGE MISS RATES")
    print("="*60)
    
    model_miss_rates = {}
    # Calculate miss rates for each model/stage combination
    for model_name, experiment_dir, model_folder in model_experiments:
        for stage in target_stages:
            model_stage_key = f"{model_folder if model_folder else model_name}_{stage}"
            model_misses = [m for m in all_missed_pages if m['model_stage'] == model_stage_key]
            total_gt_pages = sum(len(pages) for pages in ground_truth.values())
            miss_rate = len(model_misses) / total_gt_pages if total_gt_pages > 0 else 0
            model_miss_rates[model_stage_key] = miss_rate
    
    model_performance = pd.DataFrame([
        {'Model_Stage': k, 'Miss_Rate': f"{v:.3f}"} 
        for k, v in sorted(model_miss_rates.items(), key=lambda x: x[1])
    ])
    print(model_performance.to_string(index=False))
    
    print(f"\nResults saved to current directory:")
    print(f"  - 2_chapter_pages_missed_detailed_analysis.json")
    print(f"  - 2_chapter_pages_missed_by_most_models.csv")
    print(f"  - 3_chapter_pages_missed_by_all_models.csv")
    print(f"  - 4_chapter_pages_miss_rates_by_label_type.csv")


if __name__ == "__main__":
    analyze_missed_pages()