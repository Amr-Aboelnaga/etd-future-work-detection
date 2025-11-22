#!/usr/bin/env python3
"""
Calculate metrics for a subset of the ground truth data by parsing conclusion_beginnings.txt files
from all pipeline experiments and comparing against final_groundtruth_filtered.csv.

Only evaluates PDFs that are present in the ground truth subset - ignores any predictions
for PDFs not in the ground truth to avoid false positives.
"""

import os
import re
import csv
import json
from pathlib import Path
from collections import defaultdict

def parse_ground_truth_csv(csv_path):
    """Parse the ground truth CSV and create filename -> page mappings"""
    ground_truth = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            full_path = row['full_path']
            filename = os.path.basename(full_path)
            page_num = int(row['page_number'])
            
            if filename not in ground_truth:
                ground_truth[filename] = set()
            ground_truth[filename].add(page_num)
    
    return ground_truth

def parse_conclusion_beginnings_filtered(file_path, ground_truth_files):
    """
    Parse conclusion_beginnings.txt file and extract predictions, 
    but ONLY for files that exist in our ground truth subset
    """
    predictions = {}
    
    if not os.path.exists(file_path):
        return predictions
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Look for patterns like: PDF: /path/to/file.pdf - Page: 123
    pattern = r'PDF: (.+?) - Page: (\d+)'
    matches = re.findall(pattern, content)
    
    for pdf_path, page_str in matches:
        filename = os.path.basename(pdf_path)
        
        # ONLY include predictions for files in our ground truth subset
        if filename in ground_truth_files:
            page_num = int(page_str)
            
            if filename not in predictions:
                predictions[filename] = set()
            predictions[filename].add(page_num)
    
    return predictions

def calculate_metrics(ground_truth, predictions):
    """Calculate precision, recall, and F1 metrics"""
    # Only evaluate files that are in the ground truth
    all_files = set(ground_truth.keys())
    
    total_tp = 0  # True positives
    total_fp = 0  # False positives  
    total_fn = 0  # False negatives
    
    for filename in all_files:
        gt_pages = ground_truth.get(filename, set())
        pred_pages = predictions.get(filename, set())
        
        # Calculate per-file metrics
        tp = len(gt_pages & pred_pages)  # Intersection - correct predictions
        fp = len(pred_pages - gt_pages)  # Predicted but not in ground truth
        fn = len(gt_pages - pred_pages)  # In ground truth but not predicted
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # Calculate overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'total_predictions': total_tp + total_fp,
        'total_gt_pages': total_tp + total_fn
    }

def main():
    # Paths
    ground_truth_csv = "/projects/open_etds/amr_data/final_experiments/groundtruth_subset_100.csv"
    base_results_dir = "/projects/open_etds/amr_data/final_experiments/pipeline_experiment/results"
    
    # Parse ground truth
    print("Parsing ground truth CSV...")
    ground_truth = parse_ground_truth_csv(ground_truth_csv)
    ground_truth_files = set(ground_truth.keys())
    print(f"Found {len(ground_truth)} files in ground truth subset with {sum(len(pages) for pages in ground_truth.values())} total conclusion pages")
    
    # Define experiment directories and models
    experiment_dirs = {
        "llama4scout": "llama4ScoutResults/pipeline_experiments_20250907_020533_llama4:scout",
        "llama3.1-8b": "llama3.18bResults/pipeline_experiments_20250906_200708_llama3.1:8b",
        "llama3.2-3b": "llama32Results/pipeline_experiments_20250907_163551_llama3.2:3b", 
        "llama3.3": "llama33Results/pipeline_experiments_20250907_072245_llama3.3",
        "mistral-small": "mistralsmallResults/pipeline_experiments_20250906_225401_mistral_small"
    }
    
    stages = ["stage3_only", "stage1_3", "stage2_3", "full_pipeline"]
    
    results = {}
    
    # Process each model and stage
    for model_name, exp_dir in experiment_dirs.items():
        results[model_name] = {}
        print(f"\nProcessing {model_name}...")
        
        for stage in stages:
            conclusion_file = os.path.join(base_results_dir, exp_dir, stage, "conclusion_beginnings.txt")
            
            if os.path.exists(conclusion_file):
                print(f"  {stage}: ", end="")
                
                # Only get predictions for files in our ground truth subset
                predictions = parse_conclusion_beginnings_filtered(conclusion_file, ground_truth_files)
                metrics = calculate_metrics(ground_truth, predictions)
                results[model_name][stage] = metrics
                
                print(f"{metrics['total_predictions']} predictions, {metrics['total_tp']} correct (P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f})")
            else:
                print(f"  {stage}: FILE NOT FOUND")
                results[model_name][stage] = {
                    'precision': 0, 'recall': 0, 'f1': 0,
                    'total_tp': 0, 'total_fp': 0, 'total_fn': 0,
                    'total_predictions': 0, 'total_gt_pages': sum(len(pages) for pages in ground_truth.values())
                }
    
    # Generate markdown tables organized by stage
    stage_descriptions = {
        "stage3_only": "Stage 3 Only (Conclusion Detection)",
        "stage1_3": "Stage 1+3 (Layout + Conclusion Detection)",
        "stage2_3": "Stage 2+3 (Page Classification + Conclusion Detection)",
        "full_pipeline": "Full Pipeline (Layout + Classification + Conclusion Detection)"
    }
    
    total_gt_pages = sum(len(pages) for pages in ground_truth.values())
    
    markdown_content = f"# Pipeline Experiment Results by Stage (Ground Truth Subset)\n\n"
    markdown_content += f"**Ground Truth Subset:** {len(ground_truth)} files with {total_gt_pages} conclusion pages\n\n"
    markdown_content += f"Note: Only evaluates predictions for PDFs present in ground truth subset. Predictions for other PDFs are ignored.\n\n"
    
    for stage in stages:
        markdown_content += f"## {stage_descriptions[stage]}\n\n"
        markdown_content += "| Model | Precision | Recall | F1 Score | Predictions | True Positives | False Positives | False Negatives |\n"
        markdown_content += "|-------|-----------|--------|----------|-------------|----------------|-----------------|-----------------||\n"
        
        for model_name in experiment_dirs.keys():
            metrics = results[model_name][stage]
            markdown_content += f"| {model_name} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | "
            markdown_content += f"{metrics['total_predictions']} | {metrics['total_tp']} | {metrics['total_fp']} | {metrics['total_fn']} |\n"
        
        markdown_content += "\n"
    
    # Save results
    with open(os.path.join(base_results_dir, "../subset_metrics_by_stage.md"), "w") as f:
        f.write(markdown_content)
    
    with open(os.path.join(base_results_dir, "../subset_metrics_raw.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to:")
    print(f"- subset_metrics_by_stage.md")
    print(f"- subset_metrics_raw.json")
    
    # Print summary comparison
    print(f"\nSummary:")
    print(f"Ground truth subset: {len(ground_truth)} files, {total_gt_pages} conclusion pages")
    
    # Find best performing model/stage combinations
    best_f1 = 0
    best_precision = 0
    best_recall = 0
    
    for model_name in experiment_dirs.keys():
        for stage in stages:
            metrics = results[model_name][stage]
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_f1_combo = f"{model_name} - {stage}"
            if metrics['precision'] > best_precision:
                best_precision = metrics['precision']
                best_precision_combo = f"{model_name} - {stage}"
            if metrics['recall'] > best_recall:
                best_recall = metrics['recall']
                best_recall_combo = f"{model_name} - {stage}"
    
    print(f"Best F1: {best_f1:.4f} ({best_f1_combo})")
    print(f"Best Precision: {best_precision:.4f} ({best_precision_combo})")
    print(f"Best Recall: {best_recall:.4f} ({best_recall_combo})")

if __name__ == "__main__":
    main()